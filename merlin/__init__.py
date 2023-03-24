import importlib
import json
import logging
import os
import pathlib

import dotenv

from merlin.core import dataset

env_path = pathlib.Path("~/.merlinenv").expanduser()

if env_path.exists():
    dotenv.load_dotenv(env_path)

    try:
        DATA_HOME = pathlib.Path(os.environ.get("DATA_HOME"))
        ANALYSIS_HOME = pathlib.Path(os.environ.get("ANALYSIS_HOME"))
        PARAMETERS_HOME = pathlib.Path(os.environ.get("PARAMETERS_HOME"))
        ANALYSIS_PARAMETERS_HOME = PARAMETERS_HOME / "analysis"
        CODEBOOK_HOME = PARAMETERS_HOME / "codebooks"
        DATA_ORGANIZATION_HOME = PARAMETERS_HOME / "dataorganization"
        POSITION_HOME = PARAMETERS_HOME / "positions"
        MICROSCOPE_PARAMETERS_HOME = PARAMETERS_HOME / "microscope"
        FPKM_HOME = PARAMETERS_HOME / "fpkm"
        SNAKEMAKE_PARAMETERS_HOME = PARAMETERS_HOME / "snakemake"

    except TypeError:
        logging.exception(
            "MERlin environment appears corrupt. Please run 'merlin --configure .' "
            "in order to configure the environment."
        )
else:
    logging.error(
        f"Unable to find MERlin environment file at {env_path}. Please run "
        "'merlin --configure .' in order to configure the environment."
    )


def store_env(data_home: str, analysis_home: str, parameters_home: str) -> None:
    """Save the paths in the MERlin environment file."""
    with env_path.open("w") as f:
        f.write(f"DATA_HOME={data_home}\n")
        f.write(f"ANALYSIS_HOME={analysis_home}\n")
        f.write(f"PARAMETERS_HOME={parameters_home}\n")


class IncompatibleVersionError(Exception):
    """Raised on obsolete analysis folders."""


def version() -> str:
    """Return the version of MERlin."""
    import pkg_resources

    return pkg_resources.get_distribution("merlin").version


def is_compatible(test_version: str, base_version: str = None) -> bool:
    """Determine if testVersion is compatible with baseVersion.

    Args:
        test_version: the version identifier to test, as the string 'x.y.z'
            where x is the major version, y is the minor version,
            and z is the patch.
        base_version: the version to check testVersion's compatibility. If  not
            specified then the current MERlin version is used as baseVersion.
    Returns: True if testVersion are compatible, otherwise false.
    """
    if base_version is None:
        base_version = version()
    return test_version.split(".")[0] == base_version.split(".")[0]


def get_analysis_datasets(max_depth: int = 2) -> list[dataset.DataSet]:
    """Get a list of all datasets currently stored in analysis home.

    Args:
        max_depth: the directory depth to search for datasets.
    Returns: A list of the dataset objects currently within analysis home.
    """
    metadata_files = []
    for depth in range(1, max_depth + 1):
        metadata_files.extend(ANALYSIS_HOME.glob("/".join(["*"] * depth) + "/dataset.json"))

    def load_dataset(json_path: pathlib.Path) -> dataset.DataSet:
        metadata = json.loads(json_path.read_text())
        module = importlib.import_module(metadata["module"])
        task = getattr(module, metadata["class"])
        return task(metadata["dataset_name"])

    return [load_dataset(m) for m in metadata_files]
