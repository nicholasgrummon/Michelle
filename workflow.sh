#!/bin/bash
source ~/.virtual_envs/python_venv/bin/activate

mkdir -p log_files
touch log_files/all_convo_history.txt
touch log_files/present_convo_log.txt
touch log_files/prompt.txt

voice="voice_cmu_us_slt_arctic_hts"

### QUIET LOOP ###
while true;
do
    # listen for trigger
    python utils/listener.py --trigger

    # say scripted greeting
    echo "($voice) (SayText \"Hey there! Let's chat\")" | festival --pipe
    
    ### CONVERSATION LOOP ###
    while true;
    do	
        # listen until quiet
        python utils/listener.py --quiet > log_files/prompt.txt
        exit_code=$?
        
        if [ $exit_code = 1 ]; then
            echo "($voice) (SayText \"Ok, thanks for the chat.\")" | festival --pipe
            break
        fi

        # get response ollama
        echo "($voice) (SayText \"$(cat log_files/prompt.txt | ollama run michelle_v0)\")" | festival --pipe
    done
done