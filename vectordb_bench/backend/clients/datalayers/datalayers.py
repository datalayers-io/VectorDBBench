"""Wrapper around the Datalayers vector database using Arrow Flight SQL."""

import logging
import os
from contextlib import contextmanager
from typing import Any

import pyarrow as pa
from flightsql import FlightSQLClient
from flightsql.client import PreparedStatement
from pyarrow import flight

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import DatalayersConfigDict, DatalayersIndexConfig

log = logging.getLogger(__name__)

# Number of partitions to create table
DEFAULT_NUM_PARTITIONS: int = 8
# Batch size for inserting embeddings
DEFAULT_LOAD_BATCH_SIZE: int = 100


class Datalayers(VectorDB):
    """Use Datalayers Arrow Flight SQL API."""

    # FlightSQLClient and cached prepared statements are shared by the load runner
    # when a backend is marked thread-safe. Datalayers inserts must be serialized.
    thread_safe: bool = False

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
        # Use a fixed-size Arrow list to match VECTOR(dim) for embeddings.
        self._embedding_type = pa.list_(pa.float32(), self.dim)
        self._insert_stmt: PreparedStatement | None = None
        self._insert_sql: str | None = None
        self._search_stmt: PreparedStatement | None = None
        self._search_sql: str | None = None

        self.conn: FlightSQLClient | None = self._create_client()

        if drop_old:
            self._drop_table()
            self._create_db_table(dim)

        if self.conn:
            self.conn.close()
            self.conn = None
        self._close_prepared()

    @contextmanager
    def init(self):
        self.conn = self._create_client()

        try:
            yield
        finally:
            self._close_prepared()
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
            columns = [self._pk_col]
            if labels_data is not None:
                columns.append(self._label_col)
            columns.append(self._vec_col)
            placeholders = ", ".join(["?"] * len(columns))
            sql = (
                f"INSERT INTO {self.db_config['database']}.{self.table_name} "
                f"({', '.join(columns)}) VALUES ({placeholders})"
            )

            prepared_stmt = self._get_insert_stmt(sql)
            # Insert in batches
            for batch_start in range(0, len(embeddings), self.load_batch_size):
                batch_end = min(batch_start + self.load_batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]
                batch_metadata = metadata[batch_start:batch_end]
                batch_labels = labels_data[batch_start:batch_end] if labels_data is not None else None

                binding = self._make_insert_binding(
                    batch_embeddings,
                    batch_metadata,
                    batch_labels,
                )
                self._execute_prepared(prepared_stmt, binding)
                insert_count += len(batch_metadata)
            return insert_count, None
        except Exception as e:
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
            where_conditions = []
            if self._where_clause:
                where_conditions.append(self._where_clause.removeprefix("WHERE ").strip())
            query_params = self._query_parameters_clause()
            if query_params:
                where_conditions.append(query_params)
            where_clause = f"WHERE {' AND '.join(where_conditions)} " if where_conditions else ""

            # Optional refine_factor workaround for the cross-partition merge bug.
            refine_factor = os.environ.get("DL_REFINE_FACTOR")
            params_clause = ""
            if refine_factor:
                params_clause = f"WHERE parameters('refine_factor={refine_factor}') "
                if where_clause:
                    where_clause = where_clause.replace("WHERE ", "", 1)

            sql = (
                f"SELECT {self._pk_col} "
                f"FROM {self.db_config['database']}.{self.table_name} "
                f"{params_clause}{where_clause} "
                f"ORDER BY {self.search_param['metric_func']}({self._vec_col}, ?) "
                f"LIMIT {k}"
            )
            prepared_stmt = self._get_search_stmt(sql)
            binding = self._make_query_binding(query)
            result = self._execute_prepared(prepared_stmt, binding)
            if result is None or not result["rows"]:
                return []

            columns = result["columns"]
            pk_idx = columns.index(self._pk_col) if self._pk_col in columns else 0
            return [int(row[pk_idx]) for row in result["rows"]]
        except Exception as e:
            log.warning("Failed to search Datalayers table (%s), error: %s", self.table_name, e)
            return []

    def _query_parameters_clause(self) -> str:
        params = [f"{key}={value}" for key, value in self.search_param.items() if key != "metric_func"]
        if not params:
            return ""
        args = ", ".join(f"'{param}'" for param in params)
        return f"parameters({args})"

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
            # Ensure all data is on disk before building the index.
            sql = f"FLUSH TABLE {self.db_config['database']}.{self.table_name} SYNC"
            self._execute(sql)
        except Exception as e:
            log.warning("Failed to flush Datalayers table (%s), error: %s", self.table_name, e)
            raise e from None

        if self.index_param["index_type"] in ("NONE", "FLAT"):
            return

        try:
            table = f"{self.db_config['database']}.{self.table_name}"
            create_sql = (
                f"CREATE VECTOR INDEX IF NOT EXISTS `{self._index_name}` "
                f"ON {table} (`{self._vec_col}`) "
                f"WITH ({', '.join(self._index_options())})"
            )
            log.info("Datalayers create vector index with sql: %s", create_sql)
            self._execute(create_sql)

            refresh_sql = f"REFRESH INDEX `{self._index_name}` ON {table} SYNC"
            log.info("Datalayers refresh vector index with sql: %s", refresh_sql)
            self._execute(refresh_sql)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Failed to build vector index on Datalayers table (%s), error: %s",
                self.table_name,
                e,
            )
            raise e from None

    def _index_options(self) -> list[str]:
        index_options = [
            f"TYPE={self.index_param['index_type']}",
            f"DISTANCE={self.index_param['metric']}",
        ]
        option_names = {
            "num_cells": "NUM_CELLS",
            "num_sub_vectors": "NUM_SUB_VECTORS",
            "num_bits": "NUM_BITS",
            "max_level": "MAX_LEVEL",
            "m": "M",
            "ef_construction": "EF_CONSTRUCTION",
        }
        for param_name, option_name in option_names.items():
            value = self.index_param.get(param_name)
            if value is not None and value > 0:
                index_options.append(f"{option_name}={value}")
        return index_options

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

    def _execute_prepared(self, prepared_stmt: PreparedStatement, binding: pa.RecordBatch) -> dict[str, list] | None:
        assert self.conn is not None, "Connection is not initialized"

        flight_info = prepared_stmt.execute(binding)
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

    def _get_insert_stmt(self, sql: str) -> PreparedStatement:
        assert self.conn is not None, "Connection is not initialized"
        if self._insert_stmt is None or self._insert_sql != sql:
            if self._insert_stmt is not None:
                self._insert_stmt.close()
            self._insert_stmt = self.conn.prepare(sql)
            self._insert_sql = sql
        return self._insert_stmt

    def _get_search_stmt(self, sql: str) -> PreparedStatement:
        assert self.conn is not None, "Connection is not initialized"
        if self._search_stmt is None or self._search_sql != sql:
            if self._search_stmt is not None:
                self._search_stmt.close()
            self._search_stmt = self.conn.prepare(sql)
            self._search_sql = sql
        return self._search_stmt

    def _close_prepared(self) -> None:
        if self._insert_stmt is not None:
            self._insert_stmt.close()
        if self._search_stmt is not None:
            self._search_stmt.close()
        self._insert_stmt = None
        self._insert_sql = None
        self._search_stmt = None
        self._search_sql = None

    def _is_query_sql(self, sql: str) -> bool:
        stmt = sql.strip().upper()
        return stmt.startswith(("SELECT", "SHOW", "WITH"))

    def _is_update_sql(self, sql: str) -> bool:
        stmt = sql.strip().upper()
        return stmt.startswith(("INSERT", "DELETE"))

    def _make_insert_binding(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> pa.RecordBatch:
        columns = [self._pk_col]
        arrays = [pa.array(metadata, type=pa.int32())]
        if labels_data is not None:
            columns.append(self._label_col)
            arrays.append(pa.array(labels_data, type=pa.string()))
        columns.append(self._vec_col)
        arrays.append(pa.array(embeddings, type=self._embedding_type))

        return pa.RecordBatch.from_arrays(arrays, columns)

    def _make_query_binding(self, query: list[float]) -> pa.RecordBatch:
        values = pa.array(query, type=pa.float32())
        array = pa.FixedSizeListArray.from_arrays(values, type=self._embedding_type)
        return pa.RecordBatch.from_arrays([array], [self._vec_col])

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"

        try:
            self._execute(f'DROP TABLE IF EXISTS {self.db_config["database"]}.{self.table_name}')
            log.info(f"Datalayers client drop table : {self.table_name}")
        except Exception as e:
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

            sql = f"""
                CREATE TABLE IF NOT EXISTS `{self.db_config["database"]}`.`{self.table_name}` (
                    `{self._ts_col}` TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    `{self._pk_col}` INT32 NOT NULL,
                    `{self._label_col}` STRING,
                    `{self._vec_col}` VECTOR({dim}),
                    TIMESTAMP KEY(`{self._ts_col}`),
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
            log.info(f"Datalayers create table {self.table_name} with sql: {sql}")

            self._execute(sql)

        except Exception as e:
            log.warning("Failed to create Datalayers table: %s error: %s", self.table_name, e)
            raise e from None
