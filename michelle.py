import ollama
import tools
import os

PERSONALITY_PATH = '/home/ncg/Documents/Michelle/personality'
SKILLS_PATH = '/home/ncg/Documents/Michelle/skills'
class Michelle:
    def __init__(self, modelname, secondary_modelname=None, personality_dirpath=PERSONALITY_PATH, skills_dirpath=SKILLS_PATH):
        # constructor variables
        self.modelname = modelname
        self.secondary_modelname = secondary_modelname if secondary_modelname else modelname
        self.personality_dirpath = personality_dirpath
        self.skills_dirpath = skills_dirpath

        self.context = []

        # constructor actions
        ollama.pull(self.modelname) # todo: async
        ollama.pull(self.secondary_modelname) # todo: async
        os.makedirs(self.personality_dirpath, exist_ok=True)
        os.makedirs(self.skills_dirpath, exist_ok=True)
        self.load_context()


    def add_context(self, role, content):
        '''append to the context window
        role: system, user, assistant
        content: text to be appended
        '''
        self.context.append({'role': role, 'content': content})


    def clear_context(self):
        self.context.clear()


    def load_context(self):
        # load identity
        with open(os.path.join(self.personality_dirpath, 'identity.md'), 'r') as file:
            self.add_context("system", "The following defines your identity:")
            self.add_context("system", file.read())

        # load memory
        with open(os.path.join(self.personality_dirpath, 'memory.json'), 'r') as file:
            self.add_context("system", "You know the following information:")
            self.add_context("system", file.read())
        
        # load skills
        for skill in os.listdir(self.skills_dirpath):
            if skill[0] != ".":
                with open(os.path.join(self.skills_dirpath, skill, "SKILL.md")) as file:
                    self.add_context("system", file.read())

    
    def handle_toolcalls(self, message):
        if message[:8] == "!command":
            message = message[9:]                       # drop "!command" prefix
            result = tools.execute_bash(message)
            message = " ".join(message.split()[1:])     # extract message
        
        elif message[:9] == "!remember":
            message = " ".join(message.split()[1:])
            memory = ""
            while message[0] != "!":
                memory += message[0]
                message = message[1:]
            message = message[2:]
            tools.append_file("/home/ncg/Documents/Michelle/personality/memory.json", memory+"\n")
        
        return message
    

    def chat(self, stream=False):
        response = ollama.chat(self.modelname, messages=self.context, stream=stream)
        response = self.handle_toolcalls(response.message.content)
        self.add_context("assistant", response)
        return response

    # def __del__(self):
    #     ollama.delete(self.modelname)
    #     ollama.delete(self.secondary_modelname)