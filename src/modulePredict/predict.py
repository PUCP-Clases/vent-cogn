from datetime import datetime
import numpy as np
import pandas as pd
import os
import logging
import traceback
import sys
from cryptography.fernet import Fernet
#from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from sqlalchemy import create_engine
from sqlalchemy import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_json_file, get_file_key

#%% Variables
path_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
var_path = os.path.join(path_project, 'variables.json')
var = get_json_file(var_path)

path_tr_file = os.path.join(path_project, var['transformer_path'])
path_log = os.path.join(path_project, var['log_path'])
path_error = os.path.join(path_project, var['error_path'])

schema = var['schema_name_predict_output']
table = var['table_name_predict_output']  
df_clase = pd.read_csv(os.path.join(path_tr_file,'encoding_mappings.txt'), sep='\t')
#print(df_clase, df_clase.dtypes)
#numeric_columns = df_clase.select_dtypes(include=['int', 'float']).columns
#df_clase[numeric_columns] = df_clase[numeric_columns].astype(str)

periodo = datetime.now().strftime('%Y%m')
#print(periodo)

#%% Credenciales
key_file_path = os.path.join(os.path.join(path_project, var['key_path']),'filekey.key')
key = get_file_key(key_file_path)

cred_path = os.path.join(os.path.join(path_project, var['credential_path']),'credentials.json') 
creds = get_json_file(cred_path)

level = var['level_deploy']

cred_ds_name = 'engine_' + schema.lower() + '_' + level

fernet = Fernet(key)
conn_ds = fernet.decrypt(bytes(creds[cred_ds_name], 'utf-8')).decode('utf-8')

#%% Funciones
def save_predict_to_bd(df, conn, schema, table):
    """
    Esta funcion guarda la predicción en una tabla de la base de datos

    Elaborado por: Jhosua Torres - Juan Carlos Tovar

    Args:
        df: DataFrame.
        conn: conexión a la base de datos
        schema: esquema de la tabla
        table: nombre de la tabla

    Returns:
        No retorna nada
    """
    #logging.basicConfig(filename=os.path.join(path_log,'doc_entity.log'), level=logging.ERROR)
    
    dtype_df = {
        'periodo': types.INTEGER(), 
        'id_proyecto': types.INTEGER(),
        'id_modelo': types.INTEGER(),
        'nro_hoja_ruta': types.VARCHAR(length=11),
        'id_undorganica_destino_mdl': types.INTEGER(),
        'id_undorganica_destino_ocr': types.INTEGER(),
        'unidad_organica_mdl': types.VARCHAR(length=255),
        'unidad_organica_ocr': types.VARCHAR(length=255),
        'probabilidad': types.NUMERIC(precision=6, scale=3),
        'confianza_media_ocr': types.NUMERIC(precision=6, scale=3),
        'importancia_variable': types.VARCHAR(length=4000),
    }

    df['confianza_media_ocr'].replace('', np.nan, inplace=True)
    df['confianza_media_ocr'] = df['confianza_media_ocr'].astype(float).round(3)

    df['destinatario_ocr_id'].replace('', np.nan, inplace=True)
    #df['destinatario_ocr_id'] = df['destinatario_ocr_id'].astype(int)

    df = df.rename(columns={'destinatario_ocr_siglas': 'unidad_organica_ocr', 'destinatario_ocr_id' : 'id_undorganica_destino_ocr'})

    # Cargar los datos a Oracle, agregando los nuevos valores al final de la tabla
    try:
        #startTime = datetime.now()
        df.to_sql(name=table.lower(),
                con=conn,
                schema=schema,
                if_exists='append',
                index=False,
                dtype=dtype_df)
        #print('Tiempo de exportación: ',datetime.now() - startTime)
        #print('Se han insertado los nuevos registros en la tabla exitosamente.')
    except Exception as e:
        #traceback_str = traceback.format_exc()
        #logging.error(str(datetime.now()) + f" - Error in save predict NRO_HOJA_RUTA: {df.iloc[0, 0]}; Error: {str(e)}\n{traceback_str}")
        raise e

def get_features_importance(X, tfidf_vectorizer, model, feature_importance):

    words = X.tolist()
    words = words[0].split()

    feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_importance = feature_importance

    word_importances = []
    for feature, importance in zip(feature_names, feature_importance):
        if feature in words:
            word_importances.append((feature, np.round(importance,5)))

    word_importances = sorted(word_importances, key=lambda x: x[1], reverse=True)
    dict_importances = dict(word_importances)

    top=15
    first_15_values = {key: value for key, value in list(dict_importances.items())[:top]}
    
    return str(first_15_values)


def predict_direccion(df, tfidf_vectorizer, model, modelo_id):
    """
    Esta funcion realiza la predicción del campo asunto

    Elaborado por: Jhosua Torres - Juan Carlos Tovar

    Args:
        df: DataFrame.
        tfidf_vectorizer: vector
        model: modelo

    Returns:
        Retorna el valor de la predicción con la probabilidad correspondiente
    """
    logging.basicConfig(filename=os.path.join(path_log,'doc_entity.log'), level=logging.ERROR)

    try:

        if modelo_id==1:
            X = df['cl_text']
            #feature_importance = model.feature_importances_
        elif modelo_id==2:
            X = df['cl_text_ocr']
            feature_importance = model.coef_.toarray().ravel()
        #X = df['cl_text']
        X_tfidf = tfidf_vectorizer.transform(X)
        y_pred_proba = model.predict_proba(X_tfidf)

        df['id_clase_modelo'] = np.argmax(y_pred_proba,axis=1)
        #df['id_clase_modelo'] = df['id_clase_modelo'].astype('object')

        df['probabilidad'] = np.max(y_pred_proba, axis=1)
        df['probabilidad'] = df['probabilidad'].astype(float).round(3)
        #df['probabilidad'] = df['probabilidad'].astype('object')
        df['id_modelo'] = modelo_id
        #df['modelo_id'] = df['modelo_id'].astype('object')
        df['periodo'] = periodo
        df['id_proyecto'] = 1

        df = df.merge(df_clase, how='left', on='id_clase_modelo')
        #drop id_clase_modelo column
        df = df.drop(columns=['id_clase_modelo', 'dato_suficiente', 'cl_text', 'cl_text_ocr'])

        if modelo_id==2:
            df['importancia_variable'] = get_features_importance(X, tfidf_vectorizer, model, feature_importance)
        else:
            df['importancia_variable'] = ''

        save_predict_to_bd(df, conn_ds, schema, table)

        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        df[numeric_columns] = df[numeric_columns].astype(str)

        return df
    
    except Exception as e:
        df['error'] = str(e)[:180] + '...'
        traceback_str = traceback.format_exc()
        name_file = df.iloc[0, 0]
        logging.error(str(datetime.now()) + f" - Error in predict NRO_HOJA_RUTA: {name_file}; Error: {str(e)}\n{traceback_str}")
        df.to_json(os.path.join(path_error, name_file + "_error.json"), orient='records')

