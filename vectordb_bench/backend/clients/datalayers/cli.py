from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from .. import DB, IndexType
from .config import DatalayersIndexConfig


class DatalayersConnectionTypedDict(CommonTypedDict):
    password: Annotated[
        str, click.option("--password", type=str, default="public", help="DB password", show_default=True)
    ]
    host: Annotated[str, click.option("--host", type=str, default="localhost", help="DB host", show_default=True)]
    port: Annotated[int, click.option("--port", type=int, default=8360, help="DB Port", show_default=True)]
    username: Annotated[str, click.option("--username", type=str, default="admin", help="DB user", show_default=True)]
    database: Annotated[
        str, click.option("--database", type=str, help="DataBase name", default="vector_bench_db", show_default=True)
    ]


class DatalayersIndexTypedDict(CommonTypedDict):
    index_type: Annotated[
        str,
        click.option(
            "--index-type",
            type=click.Choice(
                [
                    "HNSW",
                    "FLAT",
                    "IVF_FLAT",
                    "IVF_PQ",
                    "IVF_RQ",
                    "HNSW_PQ",
                    "HNSW_RQ",
                    "IVF_HNSW",
                    "IVF_HNSW_PQ",
                    "IVF_HNSW_RQ",
                ],
                case_sensitive=False,
            ),
            default="FLAT",
            show_default=True,
            help="Index type to build",
        ),
    ]
    num_cells: Annotated[
        int,
        click.option(
            "--num-cells",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers vector index build parameter NUM_CELLS, unset = use Datalayers default",
        ),
    ]
    num_sub_vectors: Annotated[
        int,
        click.option(
            "--num-sub-vectors",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers vector index build parameter NUM_SUB_VECTORS, unset = use Datalayers default",
        ),
    ]
    num_bits: Annotated[
        int,
        click.option(
            "--num-bits",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers vector index build parameter NUM_BITS, unset = use Datalayers default",
        ),
    ]
    max_level: Annotated[
        int,
        click.option(
            "--max-level",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers vector index build parameter MAX_LEVEL, unset = use Datalayers default",
        ),
    ]
    m: Annotated[
        int,
        click.option(
            "--m",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers vector index build parameter M, unset = use Datalayers default",
        ),
    ]
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers vector index build parameter EF_CONSTRUCTION, unset = use Datalayers default",
        ),
    ]
    ef: Annotated[
        int,
        click.option(
            "--ef",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers HNSW search parameter ef, unset = use Datalayers default",
        ),
    ]
    nprobes: Annotated[
        int,
        click.option(
            "--nprobes",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers IVF search parameter nprobes, unset = use Datalayers default",
        ),
    ]
    refine_factor: Annotated[
        int,
        click.option(
            "--refine-factor",
            type=int,
            default=0,
            show_default=True,
            help="Datalayers refine factor, unset = disable exact refinement",
        ),
    ]
    filter_mode: Annotated[
        str,
        click.option(
            "--filter-mode",
            type=str,
            default="",
            show_default=True,
            help=(
                "Datalayers filtered search parameter filter_mode: "
                "auto | auto(threshold) | prefilter | iterative, unset = use Datalayers default"
            ),
        ),
    ]


_index_type_mapping = {
    "FLAT": IndexType.Flat,
    "IVF_FLAT": IndexType.IVFFlat,
    "IVF_PQ": IndexType.IVFPQ,
    "IVF_RQ": IndexType.IVF_RABITQ,
    "HNSW": IndexType.HNSW,
    "HNSW_PQ": IndexType.HNSW_PQ,
    "HNSW_RQ": IndexType.HNSW_RQ,
    "IVF_HNSW": IndexType.IVF_HNSW,
    "IVF_HNSW_PQ": IndexType.IVF_HNSW_PQ,
    "IVF_HNSW_RQ": IndexType.IVF_HNSW_RQ,
}


class DatalayersTypedDict(DatalayersConnectionTypedDict, DatalayersIndexTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(DatalayersTypedDict)
def Datalayers(**parameters: Unpack[DatalayersTypedDict]):
    from .config import DatalayersConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    parameters["index_type"] = _index_type_mapping[parameters["index_type"].upper()]

    run(
        db=DB.Datalayers,
        db_config=DatalayersConfig(
            db_label=parameters["db_label"],
            database=parameters["database"],
            username=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=DatalayersIndexConfig(
            index=parameters["index_type"],
            num_cells=parameters["num_cells"],
            num_sub_vectors=parameters["num_sub_vectors"],
            num_bits=parameters["num_bits"],
            max_level=parameters["max_level"],
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef=parameters["ef"],
            nprobes=parameters["nprobes"],
            refine_factor=parameters["refine_factor"],
            filter_mode=parameters["filter_mode"],
        ),
        **parameters,
    )
