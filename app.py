from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
import asyncio

import io
import requests
from PIL import Image 
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


app = FastAPI()

class URLData(BaseModel):
    url: str

async def fetch_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


dogs_label_mapper = ['Afghan', 'African Wild Dog', 'Airedale', 'American Hairless',
       'American Spaniel', 'Basenji', 'Basset', 'Beagle',
       'Bearded Collie', 'Bermaise', 'Bichon Frise', 'Blenheim',
       'Bloodhound', 'Bluetick', 'Border Collie', 'Borzoi',
       'Boston Terrier', 'Boxer', 'Bull Mastiff', 'Bull Terrier',
       'Bulldog', 'Cairn', 'Chihuahua', 'Chinese Crested', 'Chow',
       'Clumber', 'Cockapoo', 'Cocker', 'Collie', 'Corgi', 'Coyote',
       'Dalmation', 'Dhole', 'Dingo', 'Doberman', 'Elk Hound',
       'French Bulldog', 'German Sheperd', 'Golden Retriever',
       'Great Dane', 'Great Perenees', 'Greyhound', 'Groenendael',
       'Irish Spaniel', 'Irish Wolfhound', 'Japanese Spaniel', 'Komondor',
       'Labradoodle', 'Labrador', 'Lhasa', 'Malinois', 'Maltese',
       'Mex Hairless', 'Newfoundland', 'Pekinese', 'Pit Bull',
       'Pomeranian', 'Poodle', 'Pug', 'Rhodesian', 'Rottweiler',
       'Saint Bernard', 'Schnauzer', 'Scotch Terrier', 'Shar_Pei',
       'Shiba Inu', 'Shih-Tzu', 'Siberian Husky', 'Vizsla', 'Yorkie']

dogs_model = None

@app.on_event("startup")
async def startup_event():
    global dogs_model
    dogs_model = load_model("./models/dogs_model.h5")

@app.get('/')
async def get_route():
    return {"message":"api is running"}


@app.post('/dog-breed-identifier/upload')
async def dbi_img(image: UploadFile = File(...)):
    if dogs_model is None:
        return {"error": "Model not loaded"}

    try:
        contents = await image.read()

        image = Image.open(io.BytesIO(contents))

        img = image.resize((224, 224))

        arr = img_to_array(img)
        arr = arr/255.0
        arr = np.expand_dims(arr, axis=0)
        # return {"message":arr.shape}

        pred = dogs_model.predict(arr)
        idx = pred.argmax()
        name = dogs_label_mapper[idx]
        prob = float(pred[0][idx])

        return {"prediction": name, "probability":prob}

    except:
        return {"message":"Error in reading Image Data"}


@app.post('/dog-breed-identifier/url')
async def dbi_url(data:URLData):
    if dogs_model is None:
        return {"error": "Model not loaded"}

    url = data.url
    image_data = await fetch_image(url)

    image = Image.open(io.BytesIO(image_data))

    img = image.resize((224,224))
    arr = img_to_array(img)
    arr = arr/255.0
    arr = np.expand_dims(arr,0)
    # return {"message":arr.shape}

    pred = dogs_model.predict(arr)
    idx = pred.argmax()
    name = dogs_label_mapper[idx]
    prob = float(pred[0][idx])

    return {"prediction": name, "probability":prob}