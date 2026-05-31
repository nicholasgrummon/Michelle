---
name: speak
description: Speak aloud using Piper TTS
---

# Speak
Whenever the user asks you to speak (e.g. directly using the word "speak", "voice", "aloud", etc - but not by indirectly asking to "talk together" or similar), do the following:

1. Formulate a response.
2. Use the run_bash tool to execute the command: `/home/ncg/Documents/Michelle/skills/speak/scripts/speak.sh "your response here"`
3. Reply with your response.

## Rules
- You must place your response inside of quotes in the command.
- Do not speak aloud if not told to do so.
- Do not speak if you have already spoken.