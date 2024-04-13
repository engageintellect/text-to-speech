# text-to-speech

## Description

This is a simple text-to-speech engine using microsoft/speecht5_tts and OpenAI.

The program takes a string as input and outputs an audio file of the speech. The program can be run from the command line and takes the input string as an argument.

## Usage

To use the program, run the following command from the command line:

```bash
cd text-to-speech
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 {SCRIPT_DIR}/main.py
```

<string>Note:</strong> Replace `{SCRIPT NAME}` with the name of the script you want to run. The script will prompt you to enter the text you want to convert to speech, and will output an audio file with the speech.
