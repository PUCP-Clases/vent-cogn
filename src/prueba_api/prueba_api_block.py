#%%
from datetime import datetime
import pandas as pd
from pydantic import BaseModel
from typing import List
import requests

import sys
import os
sys.path.append(os.path.dirname((os.path.abspath(__file__))))
#print(os.path.dirname((os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor
#%% module.py Classes
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
#%% Urls de los servicios
predict_url = "http://localhost:8000/predict/" 
#%%
import os
# get absolute path
DIRPATH = os.path.dirname(os.path.realpath(__file__))

#%%
df_test_api_p1 = pd.read_excel(os.path.join(DIRPATH, '..', '..', 'data', 'prueba_api', 'df_test_api_part1.xlsx'))
df_test_api_p1 = df_test_api_p1.sample(4)
df_test_api_p2 = pd.read_excel(os.path.join(DIRPATH, '..', '..', 'data', 'prueba_api', 'df_test_api_part2.xlsx'))
df_test_api_p2 = df_test_api_p2.sample(36)
df_test_api = pd.concat([df_test_api_p1, df_test_api_p2], ignore_index=True)

#%%
df = df_test_api.copy()
df.rename(columns={'RUTA_PDF': 'PATH_FILE'}, inplace=True)
df['PATH_FILE'] = "/"+df['NRO_HOJA_RUTA'] + "/" + df['PATH_FILE']
df.columns = df.columns.str.lower()
start_time = datetime.now()
#%%
def send_request(row):
    try:
        payload = {
            "nro_hoja_ruta": row['nro_hoja_ruta'],
            "asunto": row['asunto'],
            "remitente": row['remitente'],
            "path_file": row['path_file']
        }
        json_data = [payload]
        inputs = Inputs_prec(all=json_data)
        response = requests.post(predict_url, json=inputs.dict())
        response.raise_for_status()  # Raise an exception for any non-2xx status code
        result = response.json()
        print("Predict data:", result['df_predict'])
        
    except requests.exceptions.RequestException as e:
        print("Error:", e)

#%% Test block
with ThreadPoolExecutor(max_workers=None) as executor:
    executor.map(send_request, df.to_dict('records'))

end_time = datetime.now()
print("Tiempo de Ejecución: ", end_time - start_time)