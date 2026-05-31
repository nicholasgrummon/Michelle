class ReadFile:
    tool_def = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file ONLY when explicitly asked and given a specific filepath. Do not call this speculatively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"]
            }
        }
    }
    
    def run(self, args: dict):
        path = args["path"]
        with open(path, "r") as f:
            content = f.read()
        return content  # this goes back as the tool result message


class WriteFile:
    tool_def = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content into a file when asked to do so and given a specific filepath.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    }
    
    def run(self, args: dict):
        path = args["path"]
        content = args["content"]
        with open(path, "w") as f:
            f.write(content)
        return "Done writing" # this goes back as the tool result message
    
class RunBash:
    tool_def = {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a bash terminal command when explicitly asked to do so and given a specific command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Specific command provided to you"},
                },
                "required": ["cmd"]
            }
        }
    }
    
    def run(self, args: dict):
        cmd = args["cmd"]
        import subprocess
        result = subprocess.run(cmd, capture_output=True, shell=True, text=True)
        if result.returncode == 0:
            return(result.stdout) # this goes back as the tool result message
        else:
            return "Command failed"
    

tools_list = [
    ReadFile(),
    WriteFile(),
    RunBash()
]