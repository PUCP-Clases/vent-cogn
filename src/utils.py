"""
En este script se encuentran las funciones para el procesamiento de los documentos.
"""
#%%Imports
import datetime
import json
import shutil
import traceback
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import pickle
from io import StringIO
from functools import lru_cache
import logging
import os
from tqdm import tqdm 
import time
import joblib
import json

def get_json_file(var_path):
    """
    Esta funcion carga un json y devuelve su valor en un diccionario

    Elaborado por: Jhosua Torres

    Args:
        var_path (str): La ruta del archivo JSON que se va a cargar.

    Returns:
        dict: El contenido del archivo JSON como un diccionario.
    """
    with open(var_path) as json_file:
        var = json.load(json_file)
    return var
#%%Variables
path_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

var_path = os.path.join(path_project, 'variables.json')

var = get_json_file(var_path)

path_entity = var['datalake_landing_path']
path_log = os.path.join(path_project, var['log_path'])
path_error = os.path.join(path_project, var['error_path'])

#%%Functions
@lru_cache(maxsize=128, )
def load_pickle(filename):
    """
    Esta funcion carga un archivo pickle.

    Elaborado por: Jhosua Torres

    Args:
        filename (str): ruta del archivo pickle

    Returns:
        contents (obj): objeto cargado
    """
    with open(filename, 'rb') as file: # read file
        contents = pickle.load(file) # load contents of file
    return contents

@lru_cache(maxsize=100, )
def load_vectorizer(filename):
    """
    Carga un vectorizador TF-IDF desde un archivo utilizando joblib.

    Elaborado por: Juan Carlos Tovar

    Args:
        filename (str): El nombre del archivo que contiene el vectorizador TF-IDF.

    Returns:
        object: El vectorizador TF-IDF cargado desde el archivo.
    """
    tfidf_vectorizer  = joblib.load(filename)
    return tfidf_vectorizer

def get_file_key(key_file_path):
    """
    Esta funcion carga la llave de un archivo.

    Elaborado por: Juan Carlos Tovar

    Args:
        key_file_path (str): ruta del archivo de llave

    Returns:
        key (obj): objeto de la llave
    """
    with open(key_file_path, 'rb') as filekey:
        key = filekey.read()
    return key

def get_ftp_params(ftp_url):
    """
    Esta funcion obtiene los datos de una URL FTP.

    Elaborado por: Juan Carlos Tovar

    Args:
        ftp_url (str): URL FTP

    Returns:
        username (str): nombre de usuario
        password (str): contraseña
        ftp_server (str): servidor FTP
        port (int): puerto
    """
    parsed_url = urlparse(ftp_url)
    
    # Extracting username, password, hostname, and port
    username = parsed_url.username
    password = parsed_url.password
    ftp_server = parsed_url.hostname
    port = parsed_url.port if parsed_url.port else 21
    
    return [username, password, ftp_server, port]


def return_columns():
    """
    Esta funcion crea las columnas del dataframe de salida.

    Elaborado por: Juan Carlos Tovar

    Returns:
        new_columns (list): lista con los nombres de las columnas
    """
    # create new columns
    new_columns = ['Ruta del documento'] 
    return new_columns

def output_batch_ent(data, entities):
    """
    Esta funcion crea el diccionario de salida.

    Args:
        data (pd.DataFrame): dataframe de entrada
        entities (dict): diccionario con las entidades
    
    Returns:
        final_dict (dict): diccionario de salida
    """
    results_list = []
    x = data.to_dict('index')
    df_ent = pd.DataFrame([entities])
    y = df_ent.to_dict('index')
    for i in range(len(x)):
        results_list.append({i:{'inputs': x[i], 'output':y[i]}})

    final_dict = {'results': results_list}

    return final_dict

def get_doc_entities(data):
    """
    Esta funcion obtiene las entidades de un documento y las exporta.

    Elaborado por: Juan Carlos Tovar

    Args:
        data (pd.DataFrame): dataframe de entrada

    Returns:
        dict_doc_ent (dict): diccionario con las entidades
    """
    from src.document import DOCUMENT, DOCUMENT_SCANNED_PARSER, DOCUMENT_DIGITAL_PARSER, DOCUMENT_ENTITIES
    
    dict_doc_ent = {}

    logging.basicConfig(filename=os.path.join(path_log,'doc_entity.log'), level=logging.ERROR)

    flag = False
    if len(data) == 1:
        flag = True
    try:
        for i in tqdm(range(len(data))):
            document_path = data.iloc[i, 0] 
            start_time = time.time()
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
                    #doc_ocr.plot_with_bounding_boxes(d.digital, doc_ocr.id2bbox, doc_ocr.page_start, doc_ocr.end_page)
                    if doc_ocr.doc_type != 'OTROS':
                        doc_ent = DOCUMENT_ENTITIES(id2text=doc_ocr.id2text, id2bbox=doc_ocr.id2bbox)
                        dict_doc_ent = doc_ent.entities_content
                        end_time = time.time()
                    else:
                        dict_doc_ent['fecha_extract'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    if d.digital:
                        dict_doc_ent['tipo_doc'] = "Documento Digital"
                    else:
                        dict_doc_ent['tipo_doc'] = "Documento Escaneado"
                        dict_doc_ent['confianza_media_ocr'] = doc_ocr.avg_confidence
                    dict_doc_ent['clase_doc'] = doc_ocr.doc_type
                    dict_doc_ent['nro_paginas'] = d.num_pages
                    #dict_doc_ent['nro_hoja_ruta'] = data.iloc[i, 1]
                    dict_doc_ent['tiempo_extract'] = np.round(end_time - start_time,3)

                    if data.shape[1] == 1:
                        dict_doc_ent['nro_hoja_ruta'] = "sin_nro_hoja_ruta"
                        document_path = document_path.replace('\\', '/')                  
                        #json_file = path_entity+'\\'+document_path.split('/')[-1].split('.pdf')[0]+'.json'
                    else:
                        dict_doc_ent['nro_hoja_ruta'] = data.iloc[i, 1]
                        #json_file = path_entity+'\\'+ data.iloc[i, 1]+'.json'

                    dict_doc_ent["destinatario_ocr"] = dict_doc_ent.pop("PARA")
                    dict_doc_ent["asunto_ocr"] = dict_doc_ent.pop("ASUNTO")
                    dict_doc_ent["cuerpo_ocr"] = dict_doc_ent.pop("CUERPO")
                    dict_doc_ent["remitente_ocr"] = dict_doc_ent.pop("DE")
                    dict_doc_ent["referencia_ocr"] = dict_doc_ent.pop("REF")
                    dict_doc_ent["fecha_extract"] = dict_doc_ent.pop("FECHA_EXTRACCION")

                    #d.export_json_entities(dict_doc_ent, json_file)
                else:
                    print("Error in DOCUMENT PARSER Sub Class")
                    end_time = time.time()
                    dict_doc_ent['fecha_extract'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if d.digital:
                        dict_doc_ent['tipo_doc'] = "Documento Digital"
                    else:
                        dict_doc_ent['tipo_doc'] = "Documento Escaneado"
                    dict_doc_ent['clase_doc'] = "SIN TEXTO"
                    dict_doc_ent['nro_paginas'] = d.num_pages
                    #dict_doc_ent['nro_hoja_ruta'] = data.iloc[i, 1]
                    dict_doc_ent['tiempo_extract'] = np.round(end_time - start_time,3)

                    if data.shape[1] == 1:
                        dict_doc_ent['nro_hoja_ruta'] = "sin_nro_hoja_ruta"
                        document_path = document_path.replace('\\', '/')
                        #json_file = path_entity+'\\'+document_path.split('/')[-1].split('.pdf')[0]+'.json'
                    else:
                        dict_doc_ent['nro_hoja_ruta'] = data.iloc[i, 1]
                        #json_file = path_entity+'\\'+ data.iloc[i, 1]+'.json'

                    dict_doc_ent["destinatario_ocr"] = dict_doc_ent.pop("PARA")
                    dict_doc_ent["asunto_ocr"] = dict_doc_ent.pop("ASUNTO")
                    dict_doc_ent["cuerpo_ocr"] = dict_doc_ent.pop("CUERPO")
                    dict_doc_ent["remitente_ocr"] = dict_doc_ent.pop("DE")
                    dict_doc_ent["referencia_ocr"] = dict_doc_ent.pop("REF")
                    dict_doc_ent["fecha_extract"] = dict_doc_ent.pop("FECHA_EXTRACCION")
                    
                    #d.export_json_entities(dict_doc_ent, json_file)
            else:
                print("Error in DOCUMENT Class")
            
        if flag:
            return dict_doc_ent
        
    except Exception as e:
        #traceback_str = traceback.format_exc()
        #print sysdate
        print("ERROR!!", e)
        #data['error'] = str(e)[:180] + '...'
        #logging.error(str(datetime.datetime.now()) + f" - Error processing file F: {document_path}, - NRO_HOJA_RUTA: {data.iloc[0, 1]}; Error: {str(e)}\n{traceback_str}")
        #name_pdf = os.path.basename(document_path)
        #name_pdf = name_pdf.split('.pdf')[0]
        #print("Name pdf: ", name_pdf)
        #document_path_error = os.path.join(path_error, name_pdf + "_error.pdf")
        #shutil.copy(document_path, document_path_error)
        #name_file = data.iloc[0, 1]
        #data.to_json(os.path.join(path_error, name_file + "_error.json"), orient='records')
        raise e