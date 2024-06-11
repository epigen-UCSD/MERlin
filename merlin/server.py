"""Server for distributing MERlin jobs across machines.

MERlin can be started in server mode using the parameter `--server {hostname}:{port}`. The MERlin
command should also include all the normal parameters for running the original snakemake mode.
Once the server is started, MERlin can be run in client mode on the same machine and/or other
machines that have access to the same filesystem (e.g. an NAS) where the imaging data and analysis
results are stored. The clients can be started with minimal parameters. They should have the
parameter `--client {hostname}:{port}` where `hostname` and `port` match the parameter used to
start the server. They should also specify the number of cores that should be used (`-n`) and the
number of jobs that can use the GPU simultaneously (`--gpu-jobs`). MERlin requires a `dataset`
parameter, however it is not used by the client as it will run jobs relevant to the dataset that
was given when starting the server.
"""
import curses
import datetime
import multiprocessing
import time
from xmlrpc.server import SimpleXMLRPCServer

from merlin.util.snakewriter import SnakefileGenerator, generate_shell_command


class JobGroup:
    def __init__(self, task, dependencies):
        self.name = task.analysis_name
        self.dependencies = dependencies
        self.threads = task.threads
        self.completed = 0
        self.jobs = {}
        self.finalize_job = None
        if task.is_parallel():
            for fragment in task.fragment_list:
                task.fragment = fragment
                if task.is_complete():
                    self.completed += 1
                else:
                    self.jobs[f"{self.name} {fragment}"] = generate_shell_command(task, "python", gpu_jobs=0).replace(
                        "{wildcards.i}", fragment
                    )
            if task.has_finalize_step():
                task.fragment = ""
                if task.is_complete():
                    self.completed += 1
                else:
                    self.finalize_job = generate_shell_command(task, "python", gpu_jobs=0, finalize=True)
        elif task.is_complete():
            self.completed += 1
        else:
            self.jobs = {self.name: generate_shell_command(task, "python", gpu_jobs=0)}
        self.in_progress = set()

    def next_job(self):
        for name, command in self.jobs.items():
            if name not in self.in_progress:
                self.in_progress.add(name)
                return name, command, self.threads
        if self.finalize_job:
            finalize_name = f"{self.name} Finalize"
            if finalize_name not in self.in_progress and not self.in_progress:
                self.in_progress.add(finalize_name)
                return finalize_name, self.finalize_job, self.threads
        return None

    def is_complete(self):
        return len(self.jobs) == 0 and not self.finalize_job

    @property
    def remaining(self):
        return len(self.jobs) + len(self.in_progress)

    def finish(self, jobname):
        self.in_progress.remove(jobname)
        if "Finalize" in jobname:
            self.finalize_job = None
        else:
            del self.jobs[jobname]
        self.completed += 1


def get_jobs(analysis_parameters, dataset):
    generator = SnakefileGenerator(analysis_parameters, dataset, "python")
    generator.parse_parameters()
    return {
        name: JobGroup(
            task,
            [
                getattr(task, taskname).analysis_name
                for taskname in task.dependencies
                if getattr(task, taskname).analysis_name in generator.tasks
            ],
        )
        for name, task in generator.tasks.items()
    }


class TaskServer:
    def __init__(self, messages, dataset, analysis_parameters, args):
        self.messages = messages
        self.dataset = dataset
        self.analysis_parameters = analysis_parameters
        self.args = args

    def request(self, client):
        for task in self.tasks.values():
            for dep in task.dependencies:
                if not self.tasks[dep].is_complete():
                    break
            else:
                if job := task.next_job():
                    self.messages.put(["Assigning", job[0], client])
                    return job
        return None

    def complete(self, client, jobname):
        self.tasks[jobname.split()[0]].finish(jobname)
        self.messages.put(["Completed", jobname, client])

    def start(self):
        self.messages.put(["Scanning", "Starting server"])
        self.tasks = get_jobs(self.analysis_parameters, self.dataset)
        self.messages.put(["Scanning", ""])
        self.messages.put(["Tasks", self.remaining(), self.completed()])
        hostname, port = self.args.server.split(":")
        with SimpleXMLRPCServer((hostname, int(port)), logRequests=False, allow_none=True) as server:
            server.register_function(self.request)
            server.register_function(self.complete)
            server.serve_forever()

    def remaining(self):
        return sum(task.remaining for task in self.tasks.values())

    def completed(self):
        return sum(task.completed for task in self.tasks.values())


def server_process(messages, dataset, analysis_parameters, args):
    server = TaskServer(messages, dataset, analysis_parameters, args)
    server.start()


class AnalysisStatus:
    def __init__(self):
        self.scanning = "Starting server"
        self.worker_status = {}
        self.worker_started = {}
        self.tasks = 1
        self.completed = 0

    def parse_message(self, message):
        if message[0] == "Scanning":
            self.scanning = message[1]
        elif message[0] == "Assigning":
            self.worker_status[message[2]] = message[1]
            self.worker_started[message[2]] = time.time()
        elif message[0] == "Completed":
            self.tasks -= 1
            self.completed += 1
            if message[2] in self.worker_started:
                del self.worker_started[message[2]]
            if message[2] in self.worker_status:
                del self.worker_status[message[2]]
        elif message[0] == "Tasks":
            self.tasks = message[1]
            self.completed = message[2]

    def progress(self):
        if self.scanning:
            return self.scanning
        pct = 100 * self.completed / (self.tasks + self.completed)
        return f"{self.tasks} tasks remaining, {pct:0.1f}% complete"

    def workers(self):
        if not self.worker_status:
            yield "Waiting for task requests"
        for name, status in self.worker_status.items():
            duration = datetime.timedelta(seconds=int(time.time() - self.worker_started[name]))
            yield f"{name}: {status} ({duration})"


def interface(stdscr, messages, dataset):
    stdscr.clear()
    curses.curs_set(0)
    stdscr.nodelay(True)
    clear_counter = 0
    status = AnalysisStatus()
    while True:
        while not messages.empty():
            status.parse_message(messages.get())
        if clear_counter > 10:
            stdscr.clear()
            clear_counter = 0
        clear_counter += 1
        stdscr.erase()
        stdscr.addstr(0, 0, f"MERlin running on {dataset.experiment_name}")
        stdscr.addstr(1, 0, f"Raw data path: {dataset.raw_data_path}")
        stdscr.addstr(2, 0, f"Analysis path: {dataset.analysis_path}")
        stdscr.addstr(3, 0, "-----------")
        stdscr.addstr(4, 0, status.progress())
        stdscr.addstr(5, 0, "-----------")
        for i, line in enumerate(status.workers(), start=6):
            stdscr.addstr(i, 0, line)
        stdscr.addstr(i + 2, 0, "Press Q to quit")
        stdscr.refresh()
        key = stdscr.getch()
        if key >= 0:
            key = chr(key)
            if key in {"q", "Q"}:
                break
        time.sleep(0.5)


def start_server(dataset, analysis_parameters, args):
    messages = multiprocessing.Queue()
    server_kwargs = {"messages": messages, "dataset": dataset, "analysis_parameters": analysis_parameters, "args": args}
    p = multiprocessing.Process(target=server_process, kwargs=server_kwargs)
    p.start()
    curses.wrapper(interface, messages, dataset)
    p.terminate()
    p.join()
    p.close()
