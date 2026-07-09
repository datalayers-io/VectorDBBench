from abc import abstractmethod
from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class DatalayersConfigDict(TypedDict):
    host: str
    port: int
    username: str
    password: str
    database: str


class DatalayersConfig(DBConfig):
    host: str = "localhost"
    port: int = 8360
    username: str = "admin"
    password: SecretStr = "public"
    database: str = "vector_bench_db"

    def to_dict(self) -> DatalayersConfigDict:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password.get_secret_value() if self.password else "",
            "database": self.database,
        }


class DatalayersIndexConfig(BaseModel, DBCaseConfig):

    metric_type: MetricType | None = None
    index: IndexType = IndexType.NONE
    num_cells: int = 0
    num_sub_vectors: int = 0
    num_bits: int = 0
    max_level: int = 0
    m: int = 0
    ef_construction: int = 0
    ef: int = 0
    nprobes: int = 0
    refine_factor: int = 0

    def index_param(self) -> dict:
        params = {
            "metric": self._parse_metric(),
            "index_type": self._parse_index(),
        }
        if self.num_cells > 0:
            params["num_cells"] = self.num_cells
        if self.num_sub_vectors > 0:
            params["num_sub_vectors"] = self.num_sub_vectors
        if self.num_bits > 0:
            params["num_bits"] = self.num_bits
        if self.max_level > 0:
            params["max_level"] = self.max_level
        if self.m > 0:
            params["m"] = self.m
        if self.ef_construction > 0:
            params["ef_construction"] = self.ef_construction
        return params

    def search_param(self) -> dict:
        params = {
            "metric_func": self._parse_metric_func(),
        }
        if self.ef > 0:
            params["ef"] = self.ef
        if self.nprobes > 0:
            params["nprobes"] = self.nprobes
        if self.refine_factor > 0:
            params["refine_factor"] = self.refine_factor
        return params

    def _parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "L2"
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        if self.metric_type in [MetricType.IP, MetricType.DP]:
            return "DOT"
        msg = f"Metric type {self.metric_type} is not supported for Datalayers!"
        raise ValueError(msg)

    def _parse_metric_func(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        if self.metric_type == MetricType.COSINE:
            return "cosine_distance"
        if self.metric_type in [MetricType.IP, MetricType.DP]:
            return "dot_distance"
        msg = f"Metric type {self.metric_type} is not supported for Datalayers!"
        raise ValueError(msg)

    def _parse_index(self) -> str:
        if self.index == IndexType.Flat:
            return "FLAT"
        if self.index == IndexType.IVFFlat:
            return "IVF_FLAT"
        if self.index == IndexType.IVFPQ:
            return "IVF_PQ"
        if self.index == IndexType.IVF_RABITQ:
            return "IVF_RQ"
        if self.index == IndexType.HNSW:
            return "HNSW"
        if self.index == IndexType.HNSW_RQ:
            return "HNSW_RQ"
        if self.index == IndexType.IVF_HNSW:
            return "IVF_HNSW"
        if self.index == IndexType.IVF_HNSW_RQ:
            return "IVF_HNSW_RQ"
        if self.index == IndexType.NONE:
            return "NONE"
        msg = f"Index type {self.index} is not supported for Datalayers!"
        raise ValueError(msg)
