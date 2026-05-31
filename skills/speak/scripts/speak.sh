#!/bin/bash
dialogue=$1

<<<$dialogue /home/ncg/Documents/Michelle/.venv/bin/python -m piper \
--data-dir /home/ncg/Documents/Michelle/skills/speak/resources \
-m en_US-libritts_r-medium \
--output-file - | aplay

