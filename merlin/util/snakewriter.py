import importlib
import textwrap
from typing import Any

import networkx

from merlin.core import analysistask, dataset


def expand_as_string(task: analysistask.AnalysisTask) -> str:
    """Generate the expand function for the output of a parallel analysis task."""
    filename = task.status_file("done", fragment="{g}")
    return f"expand(r'{filename}', g={list(task.fragment_list)})"


def generate_output(
    task: analysistask.AnalysisTask,
    reftask: analysistask.AnalysisTask | None = None,
    *,
    finalize: bool = False,
) -> str:
    """Generate the output string for a task.

    If `reftask` is given, then the output string is constructed to be used for the
    input of `reftask`. If both `task` and `refTask` are parallel tasks with the same
    fragment list, then only the output for a single fragment is needed for the input.
    If `task` is a parellel task but `reftask` is not parallel, or parallel with a
    different fragment list (e.g. Optimize), then the output of all fragments for `task`
    are required for input to `reftask` and the expand string is returned.
    """
    if not task.is_parallel() or finalize:
        return f"r'{task.status_file('done')}'"
    if task == reftask:
        return expand_as_string(task)
    depends_all = False
    if reftask:
        if task.is_parallel():
            if set(task.fragment_list) != set(reftask.fragment_list):
                depends_all = True
        else:
            depends_all = True
    if depends_all:
        if task.has_finalize_step():
            return f"r'{task.status_file('done')}'"
        return expand_as_string(task)
    return f"r'{task.status_file('done', fragment='{i}')}'"


def generate_input(task: analysistask.AnalysisTask, *, finalize: bool = False) -> str:
    """Generate the input string for a task."""
    if finalize:
        return generate_output(task, task)
    input_tasks = []
    for x in task.dependencies:
        attr = getattr(task, x)
        if isinstance(attr, analysistask.AnalysisTask):
            if attr.is_invisible():
                continue
            input_tasks.append(attr)
        else:
            input_tasks.extend(attr)
    return ",".join([generate_output(x, task) for x in input_tasks]) if input_tasks else ""


def generate_message(task: analysistask.AnalysisTask, *, finalize: bool = False) -> str:
    """Generate the message string for a task."""
    message = f"Finalizing {task.analysis_name}" if finalize else f"Running {task.analysis_name}"
    if task.is_parallel() and not finalize:
        message += " on {wildcards.i}"
    return message


def generate_shell_command(
    task: analysistask.AnalysisTask, python_path: str, gpu_jobs: int = None, *, finalize: bool = False
) -> str:
    """Generate the shell command for a task."""
    args = [
        python_path,
        "-m merlin",
        f"-t {task.analysis_name}",
        f"-e {task.dataSet.data_root}",
        f"-s {task.dataSet.analysis_root}",
    ]
    if gpu_jobs is None:
        gpu_jobs = 0
    args.append(f"--gpu-jobs {gpu_jobs}")
    if task.dataSet.profile:
        args.append("--profile")
    if task.is_parallel() and not finalize:
        args.append("-i {wildcards.i}")
    if task.dataSet.analysis_suffix:
        args.append(f"--suffix {task.dataSet.analysis_suffix}")
    args.append(task.dataSet.experiment_name)
    return " ".join(args)


def snakemake_rule(task: analysistask.AnalysisTask, python_path: str = "python", gpu_jobs: int = None) -> str:
    """Generate the snakemake rule for a task."""
    lines = [
        f"rule {task.analysis_name}:",
        f"  input: {generate_input(task)}",
        f"  output: {generate_output(task)}",
        f"  message: '{generate_message(task)}'",
        f"  threads: {task.threads}",
        f"  shell: r'{generate_shell_command(task, python_path, gpu_jobs)}'",
    ]
    if task.has_finalize_step():
        lines.extend(
            [
                "",
                f"rule {task.analysis_name}_Finalize:",
                f"  input: {generate_input(task, finalize=True)}",
                f"  output: {generate_output(task, finalize=True)}",
                f"  message: '{generate_message(task, finalize=True)}'",
                f"  shell: r'{generate_shell_command(task, python_path, gpu_jobs, finalize=True)}'",
            ]
        )
    return "\n".join(lines)


class SnakefileGenerator:
    def __init__(self, parameters: dict[str, Any], dataset: dataset.DataSet, python_path: str | None = None):
        self.parameters = parameters
        self.dataset = dataset
        self.python_path = python_path

    def parse_parameters(self):
        """Create a dict of analysis tasks from the parameters."""
        self.tasks = {}
        for task_dict in self.parameters["analysis_tasks"]:
            module = importlib.import_module(task_dict["module"])
            analysis_class = getattr(module, task_dict["task"])
            parameters = task_dict.get("parameters")
            name = task_dict.get("analysis_name")
            task = analysis_class(self.dataset, self.dataset.analysis_path, parameters, name, fragment="")
            self.generate_task(task)

    def generate_task(self, task) -> None:
        if not isinstance(task, analysistask.AnalysisTask):
            for t in task:
                self.generate_task(t)
        else:
            if task.analysis_name in self.tasks:
                raise Exception("Analysis tasks must have unique names. " + task.analysis_name + " is redundant.")
            # TODO This should be more careful to not overwrite an existing
            # analysis task that has already been run.
            task.save()
            if task.is_invisible():
                return  # This task does not perform any computation or produce output
            self.tasks[task.analysis_name] = task

    def identify_terminal_tasks(self, tasks: dict[str, analysistask.AnalysisTask]) -> list[str]:
        """Find the terminal tasks."""
        all_tasks = set()
        seen = set()
        for x, a in tasks.items():
            all_tasks.add(x)
            if a.has_finalize_step():
                all_tasks.add(f"{x}_Finalize")
            for d in a.dependencies:
                if isinstance(a.parameters[d], list):
                    for t in a.parameters[d]:
                        seen.add(t)
                else:
                    seen.add(a.parameters[d])
        terminal = list(all_tasks - seen)
        return [x.replace("_Finalize", "") if x.endswith("_Finalize") else x for x in terminal]

    def generate_workflow(self, gpu_jobs=None) -> str:
        """Generate a snakemake workflow for the analysis parameters.

        Returns
            the path to the generated snakemake workflow
        """
        self.parse_parameters()
        terminal_tasks = self.identify_terminal_tasks(self.tasks)
        terminal_input = ",".join([generate_output(self.tasks[x], finalize=True) for x in terminal_tasks])
        terminal_rule = f"rule all:\n  input: {terminal_input}".strip()
        task_rules = [snakemake_rule(x, self.python_path, gpu_jobs) for x in self.tasks.values()]
        snakemake_string = "\n\n".join([textwrap.dedent(terminal_rule).strip()] + task_rules)

        return self.dataset.save_workflow(snakemake_string)
