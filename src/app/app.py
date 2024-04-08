import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.utils import load_pickle, load_vectorizer
from src.utils import  output_batch_ent, return_columns, get_doc_entities
from src.module import Inputs, Inputs_prec, Inputs_pred
import pandas as pd
import numpy as np
from typing import List
from pydantic import ValidationError
import joblib

from src.modulePredict import preproc_expediente, predict_direccion

# Create an instance of FastAPI
app = FastAPI()

# get absolute path
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# set path for pickle files (models, transformers or other properties)
model_path = os.path.join(DIRPATH, '..', 'assets', 'model', 'XGC.pkl')
model_ocr_path = os.path.join(DIRPATH, '..', 'assets', 'model', 'SVC_ocr.pkl')
transformer_path = os.path.join(DIRPATH, '..', 'assets', 'transformer', 'tfidf_vectorizer.pkl')
transformer_ocr_path = os.path.join(DIRPATH, '..', 'assets', 'transformer', 'tfidf_vectorizer_ocr.pkl')
#encoding_path = os.path.join(DIRPATH, '..', 'assets', 'transformer', 'encoding_mappings.txt')

# Load the trained model, pipeline, and other properties
model = load_pickle(model_path)
model_ocr = load_pickle(model_ocr_path)
tfidf_vectorizer = load_vectorizer(transformer_path)
tfidf_vectorizer_ocr = load_vectorizer(transformer_ocr_path)
#encoding_mappings = {}

# with open(encoding_path, 'r', encoding='utf-8') as file:
#     next(file)  # Skip the header
#     for line in file:
#         encoded_label, original_label = line.strip().split('\t')
#         encoding_mappings[encoded_label] = original_label

# Configure static and template files
app.mount("/static", StaticFiles(directory="src/app/static"), name="static") # Mount static files
templates = Jinja2Templates(directory="src/app/templates") # Mount templates for HTML

# Root endpoint to serve index.html template
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})

# Health check endpoint
@app.get("/health")
def check_health():
    return {"status": "ok"}

# Model information endpoint
@app.post('/model-info')
async def model_info():
    #model_name = model.__class__.__name__ # get model name 
    model_name_1 = 'Modelo XGB de clasificación basado en texto del asunto del expediente'
    model_name_2 = 'Modelo SVC de clasificación basado en texto del cuerpo del expediente'
    #model_params = model.get_params() # get model parameters
    model_params_1 = r"{'learning_rate':'0.4', 'n_estimators':'1000'}"
    model_params_2 = r"{'C': '0.5' , 'kernel': 'linear'}"
    #features = properties['train features'] # get training feature
    features = ['NRO_HOJA_RUTA','ASUNTO_BD', 'CUERPO_PDF', 'DIRECCION_DESTINO']
    model_information =  {'model info': {
            'model name 1': model_name_1,
            'model name 2': model_name_2,
            'model 1 parameters': model_params_1,
            'model 2 parameters': model_params_2,
            'train feature': features}
            }
    return model_information # return model information
 
# Entities endpoint
@app.post('/entities')
async def entities(document_path: str):
    data = pd.DataFrame([document_path], columns=return_columns())
    dict_doc_ent = get_doc_entities(data)
    #print(dict_doc_ent)
    response = output_batch_ent(data, dict_doc_ent)
    return response

@app.post('/entities-lote')
async def entities_lote(inputs: Inputs):
    try:
        #data = pd.DataFrame([input.dict() for input in inputs.all], columns=['document_path'])
        data = inputs.return_dict_inputs()
        data = pd.DataFrame(data)

        get_doc_entities(data)
        # Do something with the DataFrame like get_doc_entities(data)
        return {"status": "success", "code": 200}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/preproc')
async def preproc_a(inputs: Inputs_prec):
    try:
        data = inputs.return_dic_inputs_prec()
        data = pd.DataFrame(data)

        preprocessed_data = preproc_expediente(data)

        if preprocessed_data is None:
            raise HTTPException(status_code=400, detail="Empty Preproc DataFrame")
        else:
            return {"preprocessed_data": preprocessed_data}
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Prediction endpoint
@app.post('/predict')
async def predict_a(inputs: Inputs_prec):
    try:
        data = inputs.return_dic_inputs_prec()
        data = pd.DataFrame(data)

        preprocessed_data = preproc_expediente(data)       

        if preprocessed_data is None:
            raise HTTPException(status_code=400, detail="Empty Preproc DataFrame")
        
        transformed_list = []

        for idx in range(len(preprocessed_data['nro_hoja_ruta'])):
            #print("idx: ", idx)
            item_dict = {}
            for key in preprocessed_data.keys():
                #print("key: ", key)
                item_dict[key] = preprocessed_data[key].iloc[idx]
            transformed_list.append(item_dict)

        inputs_p = Inputs_pred(all=transformed_list)

        data_p = inputs_p.return_dic_inputs_pred()   # Inputs_pred
        df = pd.DataFrame(data_p)
        dato_suficiente = df['dato_suficiente'][0]
        modelo_id = 0

        if dato_suficiente=='1':
            vectorizer = tfidf_vectorizer
            model_p = model
            modelo_id = 1
        elif dato_suficiente=='0':
            vectorizer = tfidf_vectorizer_ocr
            model_p = model_ocr
            modelo_id = 2

        df = predict_direccion(df, vectorizer, model_p, modelo_id)
        if df is None:
            raise HTTPException(status_code=400, detail="Empty DataFrame")
        else:
            return {"df_predict": df}
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the FastAPI application
if __name__ == '__main__':
    uvicorn.run('app:app', reload=True)
