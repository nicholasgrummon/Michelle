from michelle import Michelle
import os
import subprocess
import utils

def main():
    michelle = Michelle("llama3.1:8b")

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

    michelle.add_context("user", "Hello my name is Nicholas")
    print(michelle.chat())
    # michelle.add_context("user", "I love the color orange.")
    # print(michelle.chat())

if __name__=='__main__':
    main()