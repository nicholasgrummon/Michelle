from michelle import Michelle
import os
import subprocess
import tools
import asyncio

async def main():
    michelle = Michelle("qwen3:14b")
    await michelle.start()

    # michelle.add_context("user", "Why is the sky blue?")
    # print(michelle.context)

    # michelle.add_context("user", "Hello, my name is Nicholas.")
    # michelle.add_context("assistant", "Nice to meet you, my name is Michelle.")
    # michelle.add_context("user", "I am an engineer, and I like playing chess.")
    # michelle.add_context("assistant", "Cool! I am a writer and artist, and I like reading.")

    # michelle.save_context()

    # print(os.listdir("skills"))

    # michelle.load_context()
    # print(michelle.context)

    # result = subprocess.run(["./test.sh", "hello"], capture_output=True, text=True)
    # print(result)

    # utils.execute_bash('/home/ncg/Documents/Michelle/skills/speak/scripts/speak.sh "Hello world"')

    await michelle.add_context("user", "Run the command 'pwd'")
    # await michelle.add_context("user", "Write 'hello world' into the file '/home/ncg/Documents/Michelle/hello.txt'.")
    response = await michelle.chat(show_toolcalls=True, think=True)
    print(response.message.thinking)
    print(response.message.content)

asyncio.run(main())