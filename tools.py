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
        with open(path) as f:
            content = f.read()
        return content  # this goes back as the tool result message


class WriteFile:
    tool_def = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content into a file",
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
        with open(path) as f:
            f.write(content)
        return("done")
    
tools_list = [
    ReadFile(),
    WriteFile()
]