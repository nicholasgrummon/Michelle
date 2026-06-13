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

# Context Pruning
CONTEXT_PRUNE_THRESHOLD = 0.75   # fraction of context_size that triggers pruning
CONTEXT_PRUNE_KEEP_RECENT = 6    # most recent context messages always kept verbatim
CHARS_PER_TOKEN = 4               # rough heuristic for estimating token counts from text length
CONDENSE_PROMPT_PATH = os.path.join(SKILLS_PATH, 'summarize', 'resources', 'condense_prompt.md')

# Audio Setup
AUDIO_DIRPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio_setup')
DEVICE_INDEX   = 5          # Your microphone device, get via sounddevice.query_devices()
SAMPLE_RATE    = 16000      # Whisper expects 16 kHz
CHANNELS       = 1          # Mono
CHUNK_SECONDS  = 5          # How many seconds of audio to transcribe at once
WHISPER_MODEL  = "small"     # tiny | base | small | medium | large

class Michelle:
    def __init__(self, modelname, secondary_modelname=None, context_size=4096, keep_alive=900, personality_dirpath=PERSONALITY_PATH, skills_dirpath=SKILLS_PATH, AUDIO_DIRPATH=AUDIO_DIRPATH):
        self.modelname = modelname # primary chat model
        self.secondary_modelname = secondary_modelname if secondary_modelname else modelname # context-condenser model
        self.transcribe_model = whisper.load_model(WHISPER_MODEL)
        self.context_size = context_size
        self.keep_alive = keep_alive # integer or "forever" - length of time in seconds to keep model loaded
        self.context = []
        self.context_prelude_length = 0 # number of leading messages (identity, memory, skills) never pruned

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

        # everything above this point (identity, memory, skills) is never pruned
        self.context_prelude_length = len(self.context)


    def estimate_tokens(self, messages=None):
        '''rough estimate of the token count of a list of messages (defaults to the current context),
        using a chars-per-token heuristic since exact tokenization depends on the model
        '''
        if messages is None:
            messages = self.context
        total_chars = sum(len(message.get('content') or '') for message in messages)
        return total_chars // CHARS_PER_TOKEN


    def context_is_full(self):
        '''whether the context window is near self.context_size and should be pruned'''
        return self.estimate_tokens() > self.context_size * CONTEXT_PRUNE_THRESHOLD


    async def prune_context(self):
        '''condense the oldest prunable portion of the conversation into a short summary,
        extracting any durable facts into memory.json along the way, to free up space
        in the context window. The prelude (identity, memory, skills) and the most
        recent messages are left untouched.
        '''
        start = self.context_prelude_length
        end = len(self.context) - CONTEXT_PRUNE_KEEP_RECENT
        if end <= start:
            return # not enough conversation history to prune yet

        chunk = self.context[start:end]
        chunk_text = "\n".join(f"{message['role']}: {message['content']}" for message in chunk if message.get('content'))

        with open(CONDENSE_PROMPT_PATH, 'r') as file:
            condense_instructions = file.read()

        response = ollama.chat(model=self.secondary_modelname,
                                messages=[
                                    {"role": "system", "content": condense_instructions},
                                    {"role": "user", "content": chunk_text}
                                ],
                                think=False,
                                stream=False,
                                options={"num_ctx": self.context_size},
                                keep_alive=self.keep_alive
        )

        summary, memory_entries = self._parse_condense_response(response.message.content)

        if memory_entries:
            with open(os.path.join(self.personality_dirpath, 'memory.json'), 'a') as file:
                for entry in memory_entries:
                    file.write(entry + "\n")

        summary_message = {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
        self.context = self.context[:start] + [summary_message] + self.context[end:]
        self.context_prelude_length = start + 1


    def _parse_condense_response(self, content):
        '''split a condense_prompt response into its narrative summary and a list of
        raw memory entry lines, per the format described in condense_prompt.md
        '''
        if "MEMORY:" in content:
            summary_part, memory_part = content.split("MEMORY:", 1)
        else:
            summary_part, memory_part = content, ""

        summary = summary_part.replace("SUMMARY:", "").strip()
        memory_entries = [line.strip() for line in memory_part.strip().splitlines() if line.strip().startswith("{")]

        return summary, memory_entries


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
        if self.context_is_full():
            await self.prune_context()

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