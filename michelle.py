import os
import time
import subprocess
import threading

import ollama
import whisper

import tools.tools_file as tools_file
import audio_setup.listener as listener

# GLOBALS
# Model Setup
PERSONALITY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'personality')
SKILLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skills')
MAX_TOOL_ITERATIONS = 5

# Audio Setup
AUDIO_DIRPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio_setup')
DEVICE_INDEX   = 5          # Your microphone device, get via sounddevice.query_devices()
SAMPLE_RATE    = 16000      # Whisper expects 16 kHz
CHANNELS       = 1          # Mono
CHUNK_SECONDS  = 5          # How many seconds of audio to transcribe at once
WHISPER_MODEL  = "small"     # tiny | base | small | medium | large

class Michelle:
    def __init__(self, modelname, secondary_modelname=None, context_size=4096, keep_alive=900, personality_dirpath=PERSONALITY_PATH, skills_dirpath=SKILLS_PATH, AUDIO_DIRPATH=AUDIO_DIRPATH):
        self.modelname = modelname
        self.secondary_modelname = secondary_modelname if secondary_modelname else modelname
        self.transcribe_model = whisper.load_model(WHISPER_MODEL)
        self.context_size = context_size
        self.keep_alive = keep_alive # integer or "forever" - length of time in seconds to keep model loaded
        self.context = []

        self.this_dirpath = os.path.dirname(os.path.abspath(__file__))
        self.personality_dirpath = personality_dirpath
        self.skills_dirpath = skills_dirpath
        self.AUDIO_DIRPATH = AUDIO_DIRPATH

        self.tools = [tool.tool_def for tool in tools_file.tools_list]
        self.tool_map = {tool.tool_def["function"]["name"]: tool for tool in tools_file.tools_list}
    

    async def start(self):
        ollama.pull(self.modelname) # todo: async
        ollama.pull(self.secondary_modelname) # todo: async
        os.makedirs(self.personality_dirpath, exist_ok=True)
        os.makedirs(self.skills_dirpath, exist_ok=True)
        await self.load_context()


    async def add_context(self, role, content, tool_call_id=None):
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


    def speak(self, content):
        python_path = os.path.join(self.this_dirpath, ".venv/bin/python")
        command = (
            f'{python_path} -m piper '
            f'--data-dir {AUDIO_DIRPATH} '
            f'-m en_US-libritts_r-medium '
            '--output-file - | aplay'
        )
        
        try:
            subprocess.run(command, input=content, capture_output=True, shell=True, text=True)
        except Exception as e:
            print(e)
    
    
    def listen(self):
        listener.stop_event.clear()
        # recording thread
        rec_thread = threading.Thread(
            target=listener.record_worker,
            args=(DEVICE_INDEX, SAMPLE_RATE, CHANNELS, CHUNK_SECONDS),
            daemon=True
        )
        rec_thread.start()

        # transcription thread
        trans_thread = threading.Thread(
            target=listener.transcribe_worker,
            args=(self.transcribe_model,),
            daemon=True
        )
        trans_thread.start()

        try:
            while rec_thread.is_alive() or trans_thread.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            listener.stop_event.set()
        
        rec_thread.join(timeout=3)
        trans_thread.join(timeout=10)
        
        return listener.transcription_result
    

    async def chat(self, show_toolcalls=False, think=False, stream=False):
        iteration = 0
        while iteration < MAX_TOOL_ITERATIONS:
            response = ollama.chat(model=self.modelname,
                                    messages=self.context,
                                    tools=self.tools,
                                    think=think,
                                    stream=stream,
                                    options={"num_ctx": self.context_size},
                                    keep_alive=self.keep_alive
            )
            
            # no tool calls --> return response to user
            if not response.message.tool_calls:
                await self.add_context("assistant", response.message.content)
                return response
            
            # has tool calls --> continue loop
            for tool_call in response.message.tool_calls:
                if show_toolcalls:
                    print("\n\n",iteration, tool_call)

                tool_response = self.handle_toolcall(tool_call)
                await self.add_context("tool", tool_response)

            iteration += 1
        
        return "reached maximum iteration depth without response" # todo formulate response
    

    def conversation_loop(self):
        user_message = ""
        while "bye" not in user_message:
            user_message = self.listen()
            self.add_context("user", user_message)
            response = self.chat()
            self.speak(response.message.content)
            self.add_context("assistant", response)


    # def __del__(self):
    #     ollama.delete(self.modelname)
    #     ollama.delete(self.secondary_modelname)