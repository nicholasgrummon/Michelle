import ollama
import tools
import os

PERSONALITY_PATH = '/home/ncg/Documents/Michelle/personality'
SKILLS_PATH = '/home/ncg/Documents/Michelle/skills'
MAX_TOOL_ITERATIONS = 5

class Michelle:
    def __init__(self, modelname, secondary_modelname=None, personality_dirpath=PERSONALITY_PATH, skills_dirpath=SKILLS_PATH):
        self.modelname = modelname
        self.secondary_modelname = secondary_modelname if secondary_modelname else modelname
        self.personality_dirpath = personality_dirpath
        self.tools = [tool.tool_def for tool in tools.tools_list]
        self.tool_map = {tool.tool_def["function"]["name"]: tool for tool in tools.tools_list}
        self.skills_dirpath = skills_dirpath

        self.context = []
    

    async def start(self):
        ollama.pull(self.modelname) # todo: async
        ollama.pull(self.secondary_modelname) # todo: async
        os.makedirs(self.personality_dirpath, exist_ok=True)
        os.makedirs(self.skills_dirpath, exist_ok=True)
        await self.load_context()


    async def add_context(self, role, content):
        '''append to the context window
        role: system, user, assistant
        content: text to be appended
        '''
        self.context.append({'role': role, 'content': content})


    def clear_context(self):
        self.context.clear()


    async def load_context(self):
        # load identity
        await self.add_context("system", "The following defines your identity:")
        with open(os.path.join(self.personality_dirpath, 'identity.md'), 'r') as file:
            await self.add_context("system", file.read())

        # load memory
        await self.add_context("system", "You know the following information:")
        with open(os.path.join(self.personality_dirpath, 'memory.json'), 'r') as file:
            await self.add_context("system", file.read())
        
        # load skills
        await self.add_context("system", "You know the following skills. Skills are only additional context and are NOT tools.")
        for skill in os.listdir(self.skills_dirpath):
            if skill[0] != ".":
                with open(os.path.join(self.skills_dirpath, skill, "SKILL.md")) as file:
                    await self.add_context("system", file.read())
    

    def handle_toolcall(self, tool_call):
        name = tool_call.function.name
        args = tool_call.function.arguments
        return self.tool_map[name].run(args)


    async def chat(self, show_toolcalls=False, think=False, stream=False):
        iteration = 0
        while iteration < MAX_TOOL_ITERATIONS:
            response = ollama.chat(model=self.modelname,
                                    messages=self.context,
                                    tools=self.tools,
                                    think=think,
                                    stream=stream)
            
            # no tool calls --> return response to user
            if not response.message.tool_calls:
                await self.add_context("assistant", response)
                return response

            # has tool calls --> continue loop
            for tool_call in response.message.tool_calls:
                if show_toolcalls:
                    print(iteration, tool_call)

                tool_response = self.handle_toolcall(tool_call)
                await self.add_context("tool", tool_response)

            iteration += 1
        
        return "reached maximum iteration depth without response" # todo formulate response

    # def __del__(self):
    #     ollama.delete(self.modelname)
    #     ollama.delete(self.secondary_modelname)