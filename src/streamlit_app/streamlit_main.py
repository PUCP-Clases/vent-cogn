#%%
from datetime import datetime
import pandas as pd
from pydantic import BaseModel
from typing import List
import streamlit as st
import requests
from PIL import Image
import json
import sys
import os
import copy
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import ast
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.document import DOCUMENT, DOCUMENT_SCANNED_PARSER, DOCUMENT_DIGITAL_PARSER
from cryptography.fernet import Fernet
from src.utils import get_json_file, get_ftp_params, get_file_key


DIRPATH = os.path.dirname(os.path.realpath(__file__))
path_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
var_path = os.path.join(path_project, 'variables.json')

var = get_json_file(var_path)

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
#preprocess_url = "http://localhost:8000/preproc/"
predict_url = "http://localhost:8000/predict/" 
URL = 'http://127.0.0.1:8000/entities'

path_css = os.path.join(DIRPATH, 'style.css')
#%%
st.set_page_config(
    layout='wide',
    page_title='Ventanilla Cognitiva',
    page_icon="🤖",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Aplicación demo para la verificación del API y funcionalidades"
    }
)  
with open(path_css) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#%%

def generate_word_cloud(dictionary_data, cloud_mask):
    wordcloud = WordCloud(height=500, width=500, background_color='white',
                          stopwords=None, mask=cloud_mask,
                          contour_width=2, contour_color="blue",
                          min_font_size=10).generate_from_frequencies(dictionary_data)
    return wordcloud

#json_documents = {}
#path_dir_demo = os.path.join(DIRPATH, '..','..', 'data','prueba_api')

#files = [file for file in os.listdir(path_dir_demo) if file.endswith('.json')]

#for index, file in enumerate(files):
#    if file.lower().endswith('.json'):
#        json_documents[f'Expediente {index+1}'] = os.path.join(path_dir_demo, file)

#json_list = list(json_documents.keys())

def get_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Load the JSON data
        json_data = json.load(file)
    
    return json_data

def get_ftp_url(var):
    """
    Esta función obtiene la URL del FTP.

    Elaborado por: Juan Carlos Tovar

    Args:
        var (dict): Diccionario con las variables de configuración.

    Returns:
        str: URL del FTP.
    """
    key_file_path = os.path.join(os.path.join(path_project, var['key_path']),'filekey.key')
    key = get_file_key(key_file_path)

    cred_path = os.path.join(os.path.join(path_project, var['credential_path']),'credentials.json') 
    creds = get_json_file(cred_path)

    ftp_name = var['ftp_name']
    level = var['level_deploy']

    cred_ftp_name = 'engine_' + ftp_name + '_' + level

    fernet = Fernet(key)
    conn_ftp = fernet.decrypt(bytes(creds[cred_ftp_name], 'utf-8')).decode('utf-8')

    return conn_ftp

#%%
def get_entity(doc_path: str):

    parameters={
        'document_path':doc_path
        }
    
    response = requests.post(URL, params=parameters)
    response_text =  response.json()
    return response_text

def get_pdf_to_tmp_local(lst_p, remote_path):
    """
    Esta función descarga un archivo PDF de un servidor FTP a una carpeta temporal local.

    Elaborado por: Juan Carlos Tovar

    Args:
        lst_p (list): Lista con los parámetros de conexión al servidor FTP.
        df (str): Ruta del archivo PDF en el servidor FTP.

    Returns:
        str: Mensaje de éxito o error.
    """
    from ftplib import FTP
    import os
    try:
        # Connect to the FTP server
        ftp = FTP()
        ftp.connect(lst_p[2], lst_p[3])
        ftp.login(lst_p[0], lst_p[1])
        remote_filepath=remote_path
        
        local_filepath = os.path.join(path_project,'data/tmp')
        name_pdf_file = remote_filepath.split('/')[-1]
        local_filepath = os.path.join(local_filepath, name_pdf_file)

        with open(local_filepath, 'wb') as local_file:
            # Download the file
            ftp.retrbinary('RETR ' + remote_filepath, local_file.write)

        print(f"File downloaded successfully to {local_filepath}")
        return "OK"

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the FTP connection
        ftp.quit()

def delete_pdf_tmp_file(file_path):
    """
    Esta función elimina un archivo PDF de una carpeta temporal local.

    Elaborado por: Juan Carlos Tovar

    Args:
        file_path (str): Ruta del archivo PDF en la carpeta temporal local.

    Returns:
        str: Mensaje de éxito o error.
    """
    try:
        # Attempt to delete the file
        os.remove(file_path)
        print(f"{file_path} has been deleted successfully.")
        return "OK"
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except PermissionError:
        print(f"Permission denied to delete {file_path}.")
    except Exception as e:
        print(f"An error occurred while deleting {file_path}: {e}")
#%%
#path_doc = st.sidebar.selectbox(label='Seleccione el Expediente', options=json_list, index=None, placeholder="Seleccione...")
#path_doc = json_documents.get(path_doc)

st.sidebar.markdown('## Datos del Expediente')
path_doc = st.sidebar.text_input('Ingrese la ruta del expediente', value='', max_chars=None, key=None, type='default')
nro_hoja_ruta = st.sidebar.text_input('Ingrese el Nro. de Hoja de Ruta', value='', max_chars=None, key=None, type='default')
asunto = st.sidebar.text_input('Ingrese el Asunto', value='', max_chars=None, key=None, type='default')
remitente = st.sidebar.text_input('Ingrese el Remitente', value='', max_chars=None, key=None, type='default')

json_data = {
    'path_file': path_doc,
    'nro_hoja_ruta': nro_hoja_ruta,
    'asunto': asunto,
    'remitente': remitente
}

btn_predict = st.sidebar.button('Predecir Dirección Destino', type='primary')

if btn_predict and len(path_doc) > 0 and len(asunto) > 0:
    # Datos del expediente
    #st.markdown('#### Datos de Entrada del Expediente')
    #json_data = get_json_data(path_doc)
    #st.json(json_data)

    json_data = [json_data]

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

    # list_1 = copy.deepcopy(transformed_list)

    # for item in list_1:
    #     keys_to_delete = []
    #     for key, value in item.items():
    #         if value == 'nan' or value == '':
    #             keys_to_delete.append(key)
    #     for key in keys_to_delete:
    #         del item[key]
    
    # st.write(list_1)

    list_2 = copy.deepcopy(transformed_list)

    for item in list_2:
        keys_to_delete = []
        for key, value in item.items():
            if value == 'nan' or value == '':
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del item[key]

    c1, c2 = st.columns(2)
    with c1:
        if list_2[0]['id_modelo'] == '2':
            dictionary_data = ast.literal_eval(list_2[0]['importancia_variable'])
            cloud_mask = np.array(Image.open(os.path.join(DIRPATH,'images','nube.png')))
            wordcloud = generate_word_cloud(dictionary_data, cloud_mask)
            st.markdown('#### Palabras relevantes en el expediente')
            st.image(wordcloud.to_array(), use_column_width=True)

        probability = float(list_2[0]['probabilidad'])
        pastel_colors = ['#ff4d6d', '#ee964b', '#78c6a3', '#036666']  
        ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

        def get_color(probability):
            color_index = next(i for i, (min_val, max_val) in enumerate(ranges) if min_val <= probability <= max_val)
            return pastel_colors[color_index]
        
        bar_color = get_color(probability)

        fig, ax = plt.subplots(figsize=(7, 1))
        ax.barh(0, probability, height=0.3, color=bar_color)
        ax.set_yticks([])  # hide y-axis
        ax.spines['left'].set_visible(False)  # hide left spine
        ax.spines['right'].set_visible(False)  # hide right spine
        ax.spines['top'].set_visible(False)  # hide top spine
        ax.xaxis.set_ticks_position('bottom')  # set ticks to only bottom
        ax.spines['bottom'].set_position(('axes', 0))  # position x-axis at the bottom
        ax.set_xlim(0, 1)  # set x-axis limits
        ax.text(probability + 0.025, 0, f'{probability:.2f}', va='center', fontsize=12)  # Add text label

        st.markdown(f"#### Probabilidad de la Dirección Destino Recomendada: {list_2[0]['unidad_organica_mdl']}")
        st.pyplot(fig)

    with c2:
        st.markdown('#### Datos de la Dirección Destino Recomendada')
        st.write(list_2)
    
    if list_2[0]['id_modelo'] == '2':
        
        con_ftp = get_ftp_url(var)
        lst_p = get_ftp_params(con_ftp)
        path_remote_pdf = json_data[0]['path_file']
        result_msg_download = get_pdf_to_tmp_local(lst_p, json_data[0]['path_file'])
        name_pdf_file = path_remote_pdf.split('/')[-1]
        path_tmp = os.path.join(path_project,'data/tmp')
        path_tmp_file = os.path.join(path_tmp, name_pdf_file)
        path_doc = path_tmp_file
        if path_doc:
            document_path = path_doc
            d = DOCUMENT(pdf_path=document_path)
            if d.status:
                if d.digital:
                    doc_ocr = DOCUMENT_DIGITAL_PARSER(pdf_path=document_path, n_pages=d.num_pages)
                    if len(doc_ocr.text)==0:
                        doc_ocr = DOCUMENT_SCANNED_PARSER(pdf_path=document_path, n_pages=d.num_pages)
                        d.digital = False
                else:
                    doc_ocr = DOCUMENT_SCANNED_PARSER(pdf_path=document_path, n_pages=d.num_pages)
                    if len(doc_ocr.text)==0:
                        doc_ocr = DOCUMENT_DIGITAL_PARSER(pdf_path=document_path, n_pages=d.num_pages)
                        d.digital = True
            
                if doc_ocr.status and len(doc_ocr.text)>10:
                    doc_image = doc_ocr.plot_with_bounding_boxes(d.digital, doc_ocr.id2bbox, doc_ocr.page_start, doc_ocr.end_page)
            
            if d.digital:
                type_doc = 'Digital'
            else:
                type_doc = 'Escaneado'

            st.sidebar.write(f'**Tipo de Documento Seleccionado:** {type_doc}')

            response = get_entity(path_doc)

            c1, c2 = st.columns(2)
            if doc_image is not None:
                with c1: 
                    st.markdown('#### Documento analizado')
                    st.image(doc_image, caption='Documento con Bounding Boxes', use_column_width=1)
            with c2:
                st.markdown('#### Texto Extraido')
                st.write(response)
        else:
            st.sidebar.warning('Ingrese Ruta')


