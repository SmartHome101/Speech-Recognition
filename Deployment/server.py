import os
import sys
from fastapi import FastAPI, File, UploadFile
import uvicorn
from transformers import Wav2Vec2Processor, AutoConfig, AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import librosa
import torch
import shutil


MODEL_PATH = "../Model/wav2vec-speech-model"
TEMP_FILE = "temp.wav"
app = FastAPI()

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCTC.from_pretrained(MODEL_PATH)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    return tokenizer, model, processor

tokenizer, model, processor = load_model()

@app.get('/')
def index():
    return ('Welcome to the speech model')

@app.post('/predict')
async def predict(file:  UploadFile):
    with open(TEMP_FILE, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    speech, rate = librosa.load(TEMP_FILE, sr=16000)
    input_values = processor(speech, return_tensors = 'pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    result = tokenizer.decode(predicted_ids[0])
#    os.remove(TEMP_FILE)
    
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1',port=8000)                   
