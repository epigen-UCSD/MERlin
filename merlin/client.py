"""Client that connects to a MERlin server.

To start a client, run MERlin with the parameter `--client {hostname}:{port}` where `hostname` and
`port` match the parameter used to start the server. See server.py for more details. The number of
cores should be specified with the (`-n`) parameter, and the number of jobs that can use the GPU
simultaneously with `--gpu-jobs`. MERlin requires a `dataset` parameter, however it is not used by
the client as it will run jobs relevant to the dataset that was given when starting the server.
"""

import socket
import subprocess
import sys
import time
import xmlrpc


class Client:
    def __init__(self, args):
        self.server = xmlrpc.client.ServerProxy(f"http://{args.client}")
        self.name = socket.gethostname()
        self.jobs = {}
        self.threads_used = 0
        self.max_threads = args.core_count
        self.gpu_jobs = args.gpu_jobs

    def get_new_name(self) -> str:
        """Return a unique name for a new worker."""
        i = 1
        while f"{self.name}.{i}" in self.jobs:
            i += 1
        return f"{self.name}.{i}"

    def start_job(self) -> bool:
        try:
            name = self.get_new_name()
            if job := self.server.request(name):
                self.create_worker(job, name)
                return True
        except ConnectionRefusedError:
            return False
        return False

    def create_worker(self, job, name) -> None:
        command = job[1].replace("python", sys.executable)
        command = command.replace("--gpu-jobs 0", f"--gpu-jobs {self.gpu_jobs}")
        print(command)
        self.jobs[name] = (job[0], subprocess.Popen(command, shell=True), job[2])
        self.threads_used += job[2]
        print(f"Running {job[0]}")

    def complete_job(self, name, job) -> bool:
        try:
            self.server.complete(name, job[0])
        except ConnectionRefusedError:
            return False
        print(f"Completed {job[0]}")
        self.threads_used -= job[2]
        return True

    def poll_jobs(self):
        done = []
        for pname, job in self.jobs.items():
            status = job[1].poll()
            if status == 0:
                if self.complete_job(pname, job):
                    done.append(pname)
            elif status is not None:
                print(f"Error in {job[0]}")
                self.server.failed(pname, job[0])
                self.threads_used -= job[2]
                done.append(pname)
        for pname in done:
            del self.jobs[pname]

    def run(self) -> None:
        """The main client loop."""
        print("Client started")
        try:
            while True:
                self.poll_jobs()
                while self.threads_used <= self.max_threads and self.start_job():
                    pass
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting - waiting for jobs to finish")
            while self.jobs:
                self.poll_jobs()
                time.sleep(1)
