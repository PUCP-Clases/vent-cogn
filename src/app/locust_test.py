#%%
from locust import HttpUser, task, between
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
#%%
from pydantic import BaseModel
from typing import List

class Input(BaseModel):
    document_path: str

class Inputs(BaseModel):
    all: List[Input]

    def return_dict_inputs(
            cls,
    ):
        return [ input.dict() for input in cls.all]

class Input_prec(BaseModel):
    NRO_HOJA_RUTA: str
    ASUNTO: str
    REMITENTE: str
    PATH_FILE: str

class Inputs_prec(BaseModel):
    all: List[Input_prec]

    def return_dic_inputs_prec(
            cls,
    ):
        return [input.dict() for input in cls.all]

class Input_pred(BaseModel):
    NRO_HOJA_RUTA: str
    ASUNTO: str
    REMITENTE: str
    cl_text: str
    dato_suficiente: str
    PATH_FILE: str

class Inputs_pred(BaseModel):
    all: List[Input_pred]

    def return_dic_inputs_pred(
            cls,
    ):
        return [input.dict() for input in cls.all]
#%%
def get_sample(n):
    df_exp = pd.read_parquet(r'E:\PROJECTS\std_project\data\df_STD_2019_2024_exp_preproc_v4.parquet.gzip')
    X_test_indexes = pd.read_excel(r'E:\PROJECTS\std_project\notebooks\Predict\df_ix_predict.xlsx')
    df_exp= df_exp.loc[X_test_indexes['index'].to_list()]
    df_exp['NIVEL2_v2'] = df_exp['NIVEL2_v2'].str.strip()

    encoding_mappings = {}
    with open(r'E:\PROJECTS\std_project\src\assets\transformer\encoding_mappings_XGC.txt', 'r', encoding='utf-8') as file:
        next(file)  # Skip the header
        for line in file:
            encoded_label, original_label = line.strip().split('\t')
            encoding_mappings[encoded_label] = original_label

    df_exp['DIRECCION_DESTINO_encoded'] = df_exp['NIVEL2_v2'].map(encoding_mappings)
    
    df_exp = df_exp.sample(n, random_state=42)

    return df_exp

def get_inputs_prec(df):
    df = df[['NRO_HOJA_RUTA', 'ASUNTO', 'REMITENTE']]
    df['PATH_FILE'] = r'E:\RUTA_PDF'
    json_data = df.to_dict(orient='records')
    inputs = Inputs_prec(all=json_data)

    return inputs

def get_inputs_pred(df):
    df = df[['NRO_HOJA_RUTA', 'ASUNTO', 'REMITENTE', 'cl_text']]
    df['dato_suficiente'] = '1'
    df['PATH_FILE'] = r'E:\RUTA_PDF'
    json_data = df.to_dict(orient='records')
    inputs = Inputs_pred(all=json_data)

    return inputs
#%%
class MyUser(HttpUser):
    wait_time = between(3, 7)

    @task
    def health_check(self):
        self.client.get("/health")

    @task
    def model_info(self):
        self.client.post("/model-info")

    @task
    def preproc_asunto(self):
        n_sample = 1
        df_sample = get_sample(n_sample)
        inputs_data = get_inputs_prec(df_sample)
        self.client.post("/preproc_asunto", json=inputs_data.dict())

    @task
    def predict_asunto(self):
        n_sample = 1
        df_sample = get_sample(n_sample)
        inputs_data = get_inputs_pred(df_sample) 
        self.client.post("/predict_asunto", json=inputs_data.dict())
