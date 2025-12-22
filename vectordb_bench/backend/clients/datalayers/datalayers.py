"""Wrapper around the Datalayers vector database using Arrow Flight SQL."""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

import pyarrow.flight as flight
from flightsql import FlightSQLClient

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import DatalayersConfigDict, DatalayersIndexConfig

log = logging.getLogger(__name__)

# Number of partitions to create table
DEFAULT_NUM_PARTITIONS: int = 8
# Batch size for inserting embeddings
DEFAULT_LOAD_BATCH_SIZE: int = 100
# Default polling interval for index build task status (seconds)
DEFAULT_POLL_INTERVAL_SECONDS: int = 1

class Datalayers(VectorDB):
    """Use Datalayers Arrow Flight SQL API."""

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: DatalayersConfigDict,
        db_case_config: DatalayersIndexConfig,
        collection_name: str = "vector_bench_table",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Datalayers"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self.num_partitions = DEFAULT_NUM_PARTITIONS
        self.load_batch_size = DEFAULT_LOAD_BATCH_SIZE
        self.index_param = self.case_config.index_param()
        self.search_param = self.case_config.search_param()
        log.info("Datalayers index param: %s", self.index_param)
        log.info("Datalayers search param: %s", self.search_param)

        self._index_name = "my_vector_index"
        self._ts_col = "ts"
        self._pk_col = "id"
        self._vec_col = "embedding"
        self._label_col = "labels"
        self._where_clause = ""

        self.conn: FlightSQLClient | None = self._create_client()

        if drop_old:
            self._drop_table()
            self._create_db_table(dim)

        if self.conn:
            self.conn.close()
            self.conn = None

    @contextmanager
    def init(self):
        self.conn = self._create_client()

        try:
            yield
        finally:
            if self.conn:
                self.conn.close()
            self.conn = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        assert self.conn is not None, "Connection is not initialized"

        try:
            if labels_data is not None and len(labels_data) != len(metadata):
                raise ValueError("labels_data length must match metadata length when provided")

            insert_count = 0
            # Insert in batches
            for batch_start in range(0, len(embeddings), self.load_batch_size):
                batch_end = min(batch_start + self.load_batch_size, len(embeddings))
                values = []
                columns = [self._pk_col]
                if labels_data is not None:
                    columns.append(self._label_col)
                columns.append(self._vec_col)
                for i in range(batch_start, batch_end):
                    emb_str = "[" + ", ".join(map(str, embeddings[i])) + "]"
                    row_values = [str(metadata[i])]
                    if labels_data is not None:
                        # Escape single quotes to avoid breaking the SQL literal
                        label_val = labels_data[i].replace("'", "''")
                        row_values.append(f"'{label_val}'")
                    row_values.append(emb_str)
                    values.append(f"({', '.join(row_values)})")
                self._execute(
                    f"INSERT INTO {self.db_config['database']}.{self.table_name} "
                    f"({', '.join(columns)}) VALUES {', '.join(values)}"
                )
                insert_count += len(values)
            return insert_count, None
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Failed to insert data into Datalayers table (%s), error: %s",
                self.table_name,
                e,
            )
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        try:
            vector_literal = "[" + ", ".join(map(str, query)) + "]"
            where_clause = self._where_clause

            sql = (
                f"SELECT {self._pk_col} "
                f"FROM {self.db_config['database']}.{self.table_name} "
                f"{where_clause} "
                f"ORDER BY {self.search_param['metric_func']}({self._vec_col}, {vector_literal}) "
                f"LIMIT {k}"
            )
            result = self._execute(sql)
            if result is None or not result["rows"]:
                return []

            columns = result["columns"]
            pk_idx = columns.index(self._pk_col) if self._pk_col in columns else 0
            return [int(row[pk_idx]) for row in result["rows"]]
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to search Datalayers table (%s), error: %s", self.table_name, e)
            return []

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self._where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self._where_clause = f"WHERE {self._pk_col} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            # Align column name with filter label field for flexibility
            label_col = getattr(filters, "label_field", self._label_col)
            self._where_clause = f"WHERE {label_col} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Datalayers - {filters}"
            raise ValueError(msg)

    def optimize(self, data_size: int | None = None):
        try:
            sql = f"FLUSH TABLE {self.db_config['database']}.{self.table_name} SYNC"
            self._execute(sql)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to flush Datalayers table (%s), error: %s", self.table_name, e)
            return

        self._wait_for_index_build_tasks()

    def _wait_for_index_build_tasks(self) -> None:
        """Polls task status until build_index tasks finish."""
        while True:
            try:
                tasks = self._execute("SHOW TASKS")
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "Failed to fetch Datalayers tasks while waiting for index build: %s",
                    e,
                )
                return

            if tasks is None or not tasks["rows"]:
                log.warning("SHOW TASKS returned no data while waiting for index build")
                return

            col_map = {col.lower(): idx for idx, col in enumerate(tasks["columns"])}
            type_idx = col_map.get("type")
            running_idx = col_map.get("running")
            pending_idx = col_map.get("pending")
            if type_idx is None or running_idx is None or pending_idx is None:
                log.warning(
                    "SHOW TASKS returned unexpected columns while waiting for index build: %s",
                    tasks["columns"],
                )
                return

            build_rows = [
                row
                for row in tasks["rows"]
                if str(row[type_idx]).lower() == "build_index"
            ]
            if not build_rows:
                log.warning("SHOW TASKS does not contain build_index task info")
                return

            first_row = build_rows[0]
            try:
                running = int(first_row[running_idx])
            except Exception:
                running = 0
            try:
                pending = int(first_row[pending_idx])
            except Exception:
                pending = 0

            if running == 0 and pending == 0:
                return

            time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)

    def _create_client(self) -> FlightSQLClient:
        # Avoid proxy interference for local FlightSQL connections.
        if self.db_config["host"] in ("localhost", "127.0.0.1"):
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ.pop(key, None)
            no_proxy = os.environ.get("NO_PROXY", "")
            if "localhost" not in no_proxy or "127.0.0.1" not in no_proxy:
                combined = ",".join(filter(None, [no_proxy, "localhost", "127.0.0.1"]))
                os.environ["NO_PROXY"] = combined
                os.environ["no_proxy"] = combined

        location = f"grpc+tcp://{self.db_config['host']}:{self.db_config['port']}"
        flight_client = flight.FlightClient(location)

        headers = [
            flight_client.authenticate_basic_token(
                self.db_config["username"],
                self.db_config["password"],
            )
        ]
        headers.append((b"database", self.db_config["database"].encode("utf-8")))

        flight_sql_client = FlightSQLClient.__new__(FlightSQLClient)
        flight_sql_client.client = flight_client
        flight_sql_client.headers = headers
        flight_sql_client.features = {}
        flight_sql_client.closed = False
        return flight_sql_client

    def _execute(self, sql: str) -> dict[str, list] | None:
        assert self.conn is not None, "Connection is not initialized"

        if self._is_update_sql(sql):
            self.conn.execute_update(sql, None)
            return None

        flight_info = self.conn.execute(sql)
        if not flight_info.endpoints:
            return {"columns": [], "rows": []}

        ticket = flight_info.endpoints[0].ticket
        reader = self.conn.do_get(ticket)
        table = reader.read_all()
        if table is None:
            return None

        columns = table.schema.names
        if table.num_rows == 0:
            return {"columns": columns, "rows": []}

        column_values = [table.column(i).to_pylist() for i in range(table.num_columns)]
        rows = [list(row) for row in zip(*column_values)]
        return {"columns": columns, "rows": rows}

    def _is_query_sql(self, sql: str) -> bool:
        stmt = sql.strip().upper()
        return stmt.startswith(("SELECT", "SHOW", "WITH"))

    def _is_update_sql(self, sql: str) -> bool:
        stmt = sql.strip().upper()
        return stmt.startswith(("INSERT", "DELETE"))

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"

        try:
            self._execute(f'DROP TABLE IF EXISTS {self.db_config["database"]}.{self.table_name}')
            log.info(f"Datalayers client drop table : {self.table_name}")
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Failed to drop table %s.%s: %s",
                self.db_config["database"],
                self.table_name,
                e,
            )
            raise e from None

    def _create_db_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"

        try:
            self._execute(f'CREATE DATABASE IF NOT EXISTS {self.db_config["database"]}')

            index_statement = ""
            if self.index_param["index_type"] != "NONE" and self.index_param["index_type"] != "FLAT":
                index_statement = (
                    f"VECTOR INDEX `{self._index_name}`(`{self._vec_col}`) "
                    f"WITH (TYPE={self.index_param['index_type']}, DISTANCE={self.index_param['metric']}),"
                )

            sql = (
                f"""
                CREATE TABLE IF NOT EXISTS `{self.db_config["database"]}`.`{self.table_name}` (
                    `{self._ts_col}` TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `{self._pk_col}` INT32 NOT NULL,
                    `{self._label_col}` STRING,
                    `{self._vec_col}` VECTOR({dim}),
                    TIMESTAMP KEY(`{self._ts_col}`),
                    {index_statement}
                    PRIMARY KEY(`{self._pk_col}`, `{self._ts_col}`)
                )
                PARTITION BY HASH (`{self._pk_col}`) PARTITIONS {self.num_partitions}
                ENGINE=TimeSeries
                WITH (
                    MEMTABLE_SIZE=1024MB,
                    STORAGE_TYPE=LOCAL,
                    UPDATE_MODE=APPEND,
                    COMPACT_MODE=DISABLE
                );
                """
            )
            log.info(f"Datalayers create table {self.table_name} with sql: {sql}")

            self._execute(sql)

        except Exception as e:  # noqa: BLE001
            log.warning("Failed to create Datalayers table: %s error: %s", self.table_name, e)
            raise e from None
