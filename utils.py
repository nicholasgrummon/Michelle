import subprocess

def read_file(filepath):
    with open(filepath, "r") as file:
        return file.read()


def write_file(filepath, content):
    with open(filepath, "w") as file:
        file.write(content)


def execute_bash(command):
    result = subprocess.run(command.split(), capture_output=True, shell=True, text=True)
    return result