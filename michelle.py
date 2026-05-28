import ollama
import utils
import os

SAVE_CONTEXT_MSG = 'Summarize all messages, noting facts and who or what they describe. Refer to yourself only in second-person as "you".'

class Michelle:
    def __init__(self, modelname, secondary_modelname=None, personality_dirpath='personality', skills_dirpath='skills', voice_dirpath='voice_setup'):
        # constructor variables
        self.modelname = modelname
        self.secondary_modelname = secondary_modelname if secondary_modelname else modelname
        self.personality_dirpath = personality_dirpath
        self.skills_dirpath = skills_dirpath
        self.voice_dirpath = voice_dirpath

        self.context = []

        # constructor actions
        ollama.pull(self.modelname) # todo: async
        ollama.pull(self.secondary_modelname) # todo: async
        os.makedirs(self.personality_dirpath, exist_ok=True)
        os.makedirs(self.skills_dirpath, exist_ok=True)
        os.makedirs(self.voice_dirpath, exist_ok=True)
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
        with open(os.path.join(self.personality_dirpath, 'memory.md'), 'r') as file:
            self.add_context("system", "You know the following information:")
            self.add_context("system", file.read())
        
        # load skills
        for skill in os.listdir(self.skills_dirpath):
            if skill[0] != ".":
                with open(os.path.join(self.skills_dirpath, skill, "SKILL.md")) as file:
                    self.add_context("system", file.read())


    def save_context(self):
        self.add_context('user', SAVE_CONTEXT_MSG)
        memory = ollama.chat(self.secondary_modelname, messages=self.context)
        with open(os.path.join(self.personality_dirpath, 'memory.md'), 'w') as file:
            file.write(memory.message.content)
            file.write('\n')

    
    def handle_toolcalls(self, message):
        if message[:8] == "!command":
            message = message[9:]                       # drop "!command" prefix
            result = utils.execute_bash(message)
            message = " ".join(message.split()[1:])     # extract message
        
        return message
    

    def chat(self, stream=False):
        response = ollama.chat(self.modelname, messages=self.context, stream=stream)
        response = self.handle_toolcalls(response.message.content)
        self.add_context("assistant", response)
        return response

    # def __del__(self):
    #     ollama.delete(self.modelname)
    #     ollama.delete(self.secondary_modelname)