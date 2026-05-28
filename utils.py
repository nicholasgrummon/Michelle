import subprocess
import shlex

def read_file(filepath):
    with open(filepath, "r") as file:
        return file.read()


def write_file(filepath, content):
    with open(filepath, "w") as file:
        file.write(content)


def append_file(filepath, content):
    with open(filepath, "a") as file:
        file.write(content)


def execute_bash(command):
    result = subprocess.run(shlex.split(command), capture_output=True, text=True)
    return result