"""Main entry point for the MERlin pipeline."""
import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

import snakemake

import merlin
from merlin.core import executor
from merlin.core.dataset import MERFISHDataSet
from merlin.util import snakewriter


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Decode MERFISH data.")

    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="only generate the directory structure and do not run any analysis.",
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        help="configure MERlin environment by specifying data, analysis, and parameters directories.",
    )
    parser.add_argument("dataset", help="directory where the raw data is stored")
    parser.add_argument("-a", "--analysis-parameters", help="name of the analysis parameters file to use")
    parser.add_argument("-o", "--data-organization", help="name of the data organization file to use")
    parser.add_argument("-c", "--codebook", nargs="+", help="name of the codebook to use")
    parser.add_argument("-m", "--microscope-parameters", help="name of the microscope parameters to use")
    parser.add_argument("-p", "--positions", help="name of the position file to use")
    parser.add_argument("-n", "--core-count", type=int, help="number of cores to use for the analysis")
    parser.add_argument("--check-done", action="store_true", help="flag to only check if the analysis task is done")
    parser.add_argument(
        "-t",
        "--analysis-task",
        help="the name of the analysis task to execute. If no analysis task is provided, all tasks are executed.",
    )
    parser.add_argument(
        "-i", "--fragment-index", default="", help="the index of the fragment of the analysis task to execute"
    )
    parser.add_argument("-e", "--data-home", help="the data home directory")
    parser.add_argument("-s", "--analysis-home", help="the analysis home directory")
    parser.add_argument("-k", "--snakemake-parameters", help="the name of the snakemake parameters file")
    parser.add_argument("-f", "--fovs", help="filename containing list of FOVs to process")
    parser.add_argument("--skip", nargs="+", help="list of FOV names to omit from processing")
    parser.add_argument("--profile", action="store_true", help="profile tasks and dump to logs")

    return parser


def clean_string_arg(string: str) -> str | None:
    """Remove any single or double quotes around string."""
    return None if string is None else string.strip("'").strip('"')


def get_optional_path(string: str) -> Path | None:
    string = clean_string_arg(string)
    return Path(string) if string is not None else None


def get_input_path(prompt: str) -> str:
    """Ask user to provide a directory."""
    while True:
        path = str(input(prompt))
        if not path.startswith("s3://") and not Path(path).expanduser().exists():
            print(f"Directory {path} does not exist. Please enter a valid path.")
        else:
            return path


def configure_environment() -> None:
    """Create the merlin environment file by prompting the user."""
    data_home = get_input_path("DATA_HOME=")
    analysis_home = get_input_path("ANALYSIS_HOME=")
    parameters_home = get_input_path("PARAMETERS_HOME=")
    merlin.store_env(data_home, analysis_home, parameters_home)


def run_merlin() -> None:
    """Run the MERlin pipeline."""
    parser = build_parser()
    args, _ = parser.parse_known_args()

    if not args.analysis_task:
        print("MERlin - the MERFISH decoding pipeline")

    if args.configure:
        print("Configuring MERlin environment")
        configure_environment()
        return

    dataset = MERFISHDataSet(
        args.dataset,
        dataOrganizationName=get_optional_path(args.data_organization),
        codebookNames=args.codebook,
        microscopeParametersName=get_optional_path(args.microscope_parameters),
        positionFileName=get_optional_path(args.positions),
        dataHome=get_optional_path(args.data_home),
        analysisHome=get_optional_path(args.analysis_home),
        fovList=get_optional_path(args.fovs),
        profile=args.profile,
        skip=args.skip,
    )

    parameters_home = merlin.ANALYSIS_PARAMETERS_HOME
    # e = executor.LocalExecutor(coreCount=args.core_count)
    snakefile_path = None
    if args.analysis_parameters:
        # This is run in all cases that analysis parameters are provided
        # so that new analysis tasks are generated to match the new parameters
        with Path(parameters_home, args.analysis_parameters).open() as f:
            snakefile_path = generate_analysis_tasks_and_snakefile(dataset, f)

    if not args.generate_only:
        if args.analysis_task:
            task = dataset.load_analysis_task(args.analysis_task, args.fragment_index)
            if args.check_done:
                # checking completion creates the .done file for parallel tasks
                # where completion has not yet been checked
                if task.is_complete():
                    print(f"Task {args.analysis_task} is complete")
                else:
                    print(f"Task {args.analysis_task} is not complete")
            else:
                task.run()
        elif snakefile_path:
            snakemake_parameters = {}
            if args.snakemake_parameters:
                with Path(merlin.SNAKEMAKE_PARAMETERS_HOME, args.snakemake_parameters).open() as f:
                    snakemake_parameters = json.load(f)

            run_with_snakemake(dataset, snakefile_path, args.core_count, snakemake_parameters)


def generate_analysis_tasks_and_snakefile(dataset: MERFISHDataSet, parameters_file: TextIO) -> str:
    """Create the snakemake workflow file for the given dataset and parameters."""
    print(f"Generating analysis tasks from {parameters_file.name}")
    analysis_parameters = json.load(parameters_file)
    generator = snakewriter.SnakefileGenerator(analysis_parameters, dataset, sys.executable)
    snakefile_path = generator.generate_workflow()
    print(f"Snakefile generated at {snakefile_path}")
    return snakefile_path


def run_with_snakemake(dataset: MERFISHDataSet, snakefile_path: Path, cores: int, snakefile_parameters: dict) -> None:
    """Run the snakemake workflow."""
    print("Running MERlin pipeline through snakemake")
    snakemake.snakemake(
        snakefile_path,
        cores=cores,
        workdir=dataset.get_snakemake_path(),
        stats=snakefile_path.with_suffix(".stats"),
        lock=False,
        **snakefile_parameters,
    )
