from michelle import Michelle
import os
import subprocess
import tools.tools_file as tools_file
import asyncio

async def main():
    michelle = Michelle("qwen3:14b", context_size=15000)
    await michelle.start()

    await michelle.add_context("user", michelle.listen())
    response = await michelle.chat()

    print(response.message.content)

if __name__=='__main__':
    asyncio.run(main())