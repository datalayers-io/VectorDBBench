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
    password: Annotated[str, click.option("--password", type=str, default="public", help="DB password", show_default=True)]
    host: Annotated[str, click.option("--host", type=str, default="localhost", help="DB host", show_default=True)]
    port: Annotated[int, click.option("--port", type=int, default=8361, help="DB Port", show_default=True)]
    username: Annotated[str, click.option("--username", type=str, default="admin", help="DB user", show_default=True)]
    database: Annotated[str, click.option("--database", type=str, help="DataBase name", default="vector_bench_db", show_default=True)]


class DatalayersIndexTypedDict(CommonTypedDict):
    index_type: Annotated[
        str,
        click.option(
            "--index-type",
            type=click.Choice(
                ["HNSW", "FLAT", "IVF_FLAT", "IVF_PQ", "IVF_RQ", "IVF_HNSW"],
                case_sensitive=False,
            ),
            default="FLAT",
            show_default=True,
            help="Index type to build",
        ),
    ]

_index_type_mapping = {
    "FLAT": IndexType.Flat,
    "IVF_FLAT": IndexType.IVFFlat,
    "IVF_PQ": IndexType.IVFPQ,
    "IVF_RQ": IndexType.IVF_RABITQ,
    "HNSW": IndexType.HNSW,
    "IVF_HNSW": IndexType.IVF_HNSW,
}

class DatalayersTypedDict(DatalayersConnectionTypedDict, DatalayersIndexTypedDict):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(DatalayersTypedDict)
def Datalayers(**parameters: Unpack[DatalayersTypedDict]):
    from .config import DatalayersConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    parameters["index_type"] = _index_type_mapping[parameters["index_type"].upper()]

    run(
        db=DB.Datalayers,
        db_config=DatalayersConfig(
            db_name=parameters["database"],
            username=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=DatalayersIndexConfig(index=parameters["index_type"]),
        **parameters,
    )
