from michelle import Michelle
import os
import subprocess
import tools
import asyncio

async def main():
    michelle = Michelle("qwen3:14b", context_size=15000)
    await michelle.start()

    await michelle.add_context("user", "What is the answer to the question of life, the universe, and everything else? Speak aloud.")
    response = await michelle.chat(speaking_mode=True, show_toolcalls=True, think=False)
    print(response)

if __name__=='__main__':
    asyncio.run(main())