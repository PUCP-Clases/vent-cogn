#%%
from datetime import datetime
import pandas as pd
from pydantic import BaseModel
from typing import List
import requests
import json
import sys
#%% Clases de module.py
class Input(BaseModel):
    document_path: str

class Inputs(BaseModel):
    all: List[Input]

    def return_dict_inputs(
            cls,
    ):
        return [ input.dict() for input in cls.all]

class Input_prec(BaseModel):
    nro_hoja_ruta: str
    asunto: str
    remitente: str
    path_file: str

class Inputs_prec(BaseModel):
    all: List[Input_prec]

    def return_dic_inputs_prec(
            cls,
    ):
        return [input.dict() for input in cls.all]

class Input_pred(BaseModel):
    nro_hoja_ruta: str
    cl_text: str
    cl_text_ocr: str
    confianza_media_ocr: str
    dato_suficiente: str
    destinatario_ocr_siglas: str
    destinatario_ocr_id: str

class Inputs_pred(BaseModel):
    all: List[Input_pred]

    def return_dic_inputs_pred(
            cls,
    ):
        return [input.dict() for input in cls.all]
#%%
# Urls de los servicios
predict_url = "http://localhost:8000/predict/" 
#%%
if len(sys.argv) < 2:
    print("Usage: python script.py <json_file>")
    sys.exit(1)

# Extract the file path from the command-line arguments
file_path = sys.argv[1]
start_time = datetime.now()
with open(file_path, 'r', encoding='utf-8') as file:
    # Load the JSON data
    json_data = json.load(file)
print("=====================================")
print("Datos de entrada")
print("=====================================")
for key, value in json_data.items():
    print(f"{key} : {value}")
json_data = [json_data]
# json_data = [{'path_file': '/2022-061347/0613472022010037000525168937390202205101838452.pdf',
#   'nro_hoja_ruta': '2022-061347',
#   'asunto': 'Evaluación Presupuestal Anual 2021 del Pliego 068: Procuraduría General del Estado',
#   'remitente': 'PROCURADURIA GENERAL DEL ESTADO'}]
#%%
# API Preprocesamiento
inputs = Inputs_prec(all=json_data)
response = requests.post(predict_url, json=inputs.dict())

end_time = datetime.now()

preprocessed_data = response.json()['df_predict']
transformed_list = []

for idx in range(len(preprocessed_data['nro_hoja_ruta'])):
    item_dict = {}
    for key in preprocessed_data.keys():
        item_dict[key] = preprocessed_data[key][str(idx)]
    transformed_list.append(item_dict)

print("=====================================")
print("Resultados de API: Predicción") 
print("=====================================")
print("Tiempo Total de ejecución: ", end_time - start_time)
for item in transformed_list:
    for key, value in item.items():
        if value != 'nan' and value !='':
            print(f"{key} : {value}")