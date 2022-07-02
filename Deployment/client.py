import os
import sys
import speech_recognition as sr
import requests
import json
from pathlib import Path
module_path = str(Path.cwd())


URL = 'http://127.0.0.1:8000/predict'
TEMP_FILE = 'temp.wav'

# os.close(sys.stderr.fileno())

def save_wave_file(audio):
    with open(TEMP_FILE, "wb") as f:
        f.write(audio.get_wav_data())

def recognize():
    with open(TEMP_FILE, 'rb') as file:
        files = {'file': file}
        response = requests.post(URL, files=files)

    return response.text


if __name__ == '__main__':
    # obtain audio from the microphone
    r = sr.Recognizer()

    while True:
        r = sr.Recognizer()

        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
            save_wave_file(audio)

        text = recognize()
#        os.remove(TEMP_FILE)
        print(text)

