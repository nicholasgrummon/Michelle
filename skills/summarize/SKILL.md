---
name: summarize
description: Write facts, events, or situations to memory
---

# Summarize
Whenever information is provided to you:

1. Determine if the information is substantive and concrete. If not, stop here.
2. Determine if the information is new to you. If not new, stop here.
3. Determine if the information conflicts with what you already know. If it does, ask for clarification and stop here.
4. Determine if the entity described by the information is a person.
- user: your chat partner
- assistant: yourself (Michelle)
- another person: identify by role/relationship (e.g. "user's mother", etc)
5. If the information describes a person, determine if the information is semantic or episodic:
- semantic: facts, name, age, occupation, etc
- episodic: stories, interactions, news and updates, etc
6. If the entity described by the information is not a person, determine if it is a place, object, or something else and its name.
7. If the information describes a place or object, determine if the information is depictive or situational
- depictive: describes the appearance, functionality, or attributes.
- situational: describes how the entity is related to the conversation.
8. Record the information in the following json format: `{"entity":<role or name>, "category":<episodic, semantic, depictive, situational>, "information":<enter information here>}`
9. Use a tool to write the json format information into the file: "/home/ncg/Documents/Michelle/personality/memory1.json"

# Rules
- Ask for clarification about conflicting information.
- Format the information exactly as indicated, replacing what is inside angle brackets ("<>") with your input.
- The person entity should only be one of "user", "assistant", or <role>.
- If information describes multiple entities (e.g. a person and a place), pick the primary entity.
- The information you record should be as brief as possible. Avoid complete sentences.
- Conclude the json format with a single comma (","). Avoid extra curly brackets ("{}")