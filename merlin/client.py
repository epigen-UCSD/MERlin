import socket
import subprocess
import sys
import time
import xmlrpc


def get_new_name(name, jobs):
    i = 1
    while f"{name}.{i}" in jobs:
        i += 1
    return f"{name}.{i}"


def client(args):
    server = xmlrpc.client.ServerProxy(f"http://{args.client}")
    name = socket.gethostname()
    jobs = {}
    threads_used = 0
    while True:
        done = []
        for pname, job in jobs.items():
            status = job[1].poll()
            if status == 0:
                print(f"Completed {job[0]}")
                server.complete(pname, job[0])
                threads_used -= job[2]
                done.append(pname)
            elif status is not None:
                print(f"Error in {job[0]}")
                server.failed(pname, job[0])
                threads_used -= job[2]
                done.append(pname)
        for pname in done:
            del jobs[pname]
        if threads_used <= args.core_count:
            newname = get_new_name(name, jobs)
            if job := server.request(newname):
                command = job[1].replace("python", sys.executable)
                command = command.replace("--gpu-jobs 0", f"--gpu-jobs {args.gpu_jobs}")
                print(command)
                jobs[newname] = (job[0], subprocess.Popen(command, shell=True), job[2])
                threads_used += job[2]
                print(f"Running {job[0]}")
        time.sleep(1)
