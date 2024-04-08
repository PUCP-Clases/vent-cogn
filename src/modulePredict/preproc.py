#%% Imports
from datetime import datetime
import pandas as pd
import os
import sys
from cryptography.fernet import Fernet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import get_doc_entities, get_json_file, get_ftp_params, get_file_key

import numpy as np
import spacy

import nltk 
import os
import re
import sys
import warnings
import logging
import traceback

from nltk.corpus import stopwords
import json
from unidecode import unidecode

from nltk import download

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
#%%
nlp = spacy.load('es_core_news_sm')

# Check if 'punkt' is already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' data already exists.")
except Exception as e:
    # Download 'punkt'
    print("NLTK 'punkt' downloading")
    download('punkt')
    print("NLTK 'punkt' downloaded")

# Check if 'stopwords' is already downloaded
try:
    nltk.data.find('corpora/stopwords')
    print("NLTK 'stopwords' data already exists.")
except Exception as e:
    print("NLTK 'stopwords' downloading")
    download('stopwords')
    print("NLTK 'stopwords' downloaded")
#%%
path_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

var_path = os.path.join(path_project, 'variables.json')

var = get_json_file(var_path)

path_file = os.path.join(path_project, var['transformer_path'])

path_log = os.path.join(path_project, var['log_path'])
path_error = os.path.join(path_project, var['error_path'])

df_direcciones_mef = pd.read_excel(os.path.join(path_file,'direccion_mef.xlsx'))
df_direcciones_mef["destinatario_ocr_id"] = df_direcciones_mef["destinatario_ocr_id"].astype(str)
df_asunto_no_suficiente = pd.read_excel(os.path.join(path_file,'asunto_BD_no_suficiente.xlsx'))
df_asunto_no_suficiente = df_asunto_no_suficiente[df_asunto_no_suficiente['n_words']>1]
list_asunto_no_sufiente = df_asunto_no_suficiente['cl_text'].to_list()

path_entity = var['datalake_landing_path']


custom_stopwords = ["rtf","jrc","deg","N","Y","A","No",".:","05","E","I","D.S","5","31","O","9","1","2","n","d","s","r","m","i","u","ii","l","p","b",".",
        'N', '.', '1', '2', 'I', '3', '4', '5', '9', '6', 'S', '7', '(', 'B', '%', '8', 'C', 'D', 'F', 'A', ':', '?', 'P', 'R', ')', 'L', '$', 'H', '0', ';', ',', 'Q', 'V', '-', 'K', '&', 'T', 'J', '@', 'M', 'G', '+', '=', 'X', "'", 'O', 'o', '{', '!',
        '`', 'U', 'E', '*', '|', '[', 'a', 'Y', '<', 'W', '>', ']', '/', 'c', '#', 'Z', 'r', '}', '01',  '31', 'II', 'S/', '03', '02', '27', '19', '10', '15', '04', '21', 'Na', '05', '13', '18', '30', '09', '12', '11',  '06', '25', '22', '20', '17', '07', '16', '23', '08', '14', 'DU', 'PI', '26', '28', '24', 'N.', 'D.', 'IV', 'EX', 'TU', '29', 'SG', "''", '90', 'MD', 'AF', 'GL',
        'AV', 'JR', 'A.', 'AS', 'TR', 'SP', 'RM', 'AM', 'PS', 'UP', 'GN', 'TO', 'FE', 'AD', 'C.', 'RO', 'PM', 'US', 'DM', 'PP', '50', 'PL', 'RD', 'S.', 'MP', 'UF', '65', 'EF', 'TC', 'CD', 'E.',
        '00', 'RB', 'KV', 'SA', '48', 'U.', 'T6', 'MC', 'VS', 'N|', '36', '42', 'OF', '35', 'CP', '37', '3D', 'ID', 'R.', 'GM', '40', 'IN', 'RE', '97', '32', 'CR', 'EL', 'IE', '--', 'RP', 'ON', 'UA',
        'MN', "'S", 'DG', 'CH', '34', 'M.', 'L.', 'AI', 'IX', '60', '33', 'LA', '45', '53', 'V.', 'KM', '41', 'IP', '43', '55', 'PR', '47', 'DR', 'OA', 'BI', '38', 'LC', '2-', '70', '39', 'B.', 'MZ',
        'OC', 'DP', '54', '44', '``', 'CF', 'CM', 'PC', 'VI', 'CO', '62', '1o', 'M2', 'CA', 'IM', '98', 'AP', 'FF', '49', '61', 'GC', 'I.', '46', '56', '/O', '52', '2o', 'RA', '51', '58', 'UO', '63',
        'DN', 'TP', 'CS', '99', 'NA', '72', '59', 'PJ', '57', 'P.', 'MM', 'IC', 'XI', '88', 'DC', 'MB', 'SN', '64', '75', '80', '68', 'EN', 'S2', 'LP', 'F.', 'TF', '66', '85', '67', 'LT', 'ET', 'DV',
        '83', '5o', '4-', 'EP', '86', '79', 'PG', '74', '7A', 'S3', 'RG', 'RU', 'CI', 'F3', 'UX', '78', '73', '84', '69', '81', '71', 'SC', '77', '82', '95', '96', '87', '3-',
        'CV', 'I-', '93', 'SE', 'XV', 'AN', 'DE', 'LS', '1-', '76', 'CE', 'EM', 'AC', 'B5', 'H.', 'OR', 'SK', '9o', '6o', 'A-', 'PO', 'UD', '92', 'HP', '1A', 'YT', 'TV', 'G.', 'J.', 'OG', 'IS', 'NO', 'PA', 'OP', '-A', 'QE', 'BM', '91', 'F4', 'RS',
        'AA', 'PH', 'CN', 'CC', 'ED', 'XX', 'LI', '5-', '9-', 'F1', 'EB', 'HH', '94', 'DZ', '89', 'E-', 'DO', 'RC', 'F2', 'KA', 'BY', 'C1', 'DI', 'ES', 'CL', 'RJ', 'QU', 'IA', '8A', '4o', 'RR', 'EC',
        'T.', 'PB', 'RL', 'AR', 'EE', 'JP', 'MS', '5A', 'JF', 'ML', 'C3', 'UK', 'A4', 'B-', 'FI', 'GS', 'C4', '6-', 'AH', 'UN', 'YO', 'IT', 'PU', 'Y/', '3o', 'RH', 'SL', 'C2', 'GT', '5B', 'MA', 'KG',
        'SM', 'N-', 'GA', 'UC', 'F-', 'BE', 'A2', '-I', 'K.', 'NS', 'OS', 'AE', 'TA', 'PD', 'QA', 'CG', 'PT', 'AT', '-O', 'F5', 'A1', 'ST', 'GE', 'G2', 'MG', 'NE', 'FP', 'A3', '-2', 'EY', 'PN', 'EA',
        'UT', 'EU', 'SS', 'OK', '8o', 'BX', 'YN', 'CT', 'RN', 'R-', 'EG', 'Q.', '3A', 'WV', 'TB', '7o', 'OO', 'ER', '7C', 'FR', 'N0', '-P', '9a', 'LL', '..', 'NN', '0-', 'RI', 'AY', '.-', '8-', 'UY', 'Y=', 'SD', 'LX', 'JA', 'SO', 'AL', 'UR', '-1', 'JL', 'YG', '1B', 'YC', 'VA', 'UZ', 'S7', 'TG', 'BD',
        'KU', 'JW', '9C', '.N', '6A', 'O.', 'N1', 'VE', 'PY', 'FT', 'EI', '-5', 'E1', 'W.', 'C-', 'AB', 'G1', 'JJ', 'RT', 'DF', 'O1', 'A5', 'LU', '3N', 'BS', 'SB', 'OL', '7-', 'EO', 'HU', 'JM', 'E3',
        '-4', 'S1', 'HO', 'ZR', '4E', '1a', '3B', 'LY', 'VR', 'O2', 'Z.', '7B', 'XJ', 'BV', 'B/', '4B', 'UV', 'MX', 'EK', 'M1', 'AK', 'M3', 'B2', 'AG', 'GF', '.E', 'RV', 'JC', 'NU', 'S5', 'JB', 'KW',
        'FU', '6B', 'F9', '1Y', 'EJ', 'GB', '/A', 'HV', 'VV', 'GH', 'X.', 'P5', '-T', 'UM', 'MY', 'VP', '2D', 'VM', 'YU', 'KB', 'NC', '.1', 'JE', 'HZ', 'I3', 'SW', '8a', 'OD', 'O9', 'B1', '.2', '5U',
        'NV', 'UL', 'AU', 'V-', 'E6', 'YY', 'A6', '.Y', 'IR', 'MR', 'D1', 'E5', 'E2', 'R1', 'V2', 'OE', '2a', 'D9', 'YP', 'MW', 'GI', 'LD', 'JU', '.A', 'E4', 'HD', 'UI', 'NR', 'ZA', 'FN', 'PF', 'PV',
        'N3', 'HM', 'HT', 'BB', 'P1', '.U', 'SY', 'H1', 'I5', 'T4', '3C', 'DT', 'LV', 'C7', 'JD', 'T2', '.8', '-M', 'H2', 'MH', 'JN', 'BF', 'B3', '1/', '3O', 'P9', '-3', 'P-', 'MV', '4a', 'R8', 'LE',
        ':1', 'GO', 'FL', 'O4', '-N', 'HA', '/R', 'Y-', 'XP', 'LG', 'T-', '-R', '5E', 'D2', 'OT', 'YR', 'N2', '-F', '/S', 'C5', '-C', 'YD', 'FA', 'M6', 'O5', 'A7', 'WE', 'NB', '8B', 'K1', 'EV', 'Y3',
        'Mo', '1O', 'C9', 'AQ', 'S4', '3a', 'OM', 'M-', 'BC', 'O3', 'O8', 'MO', '8C', 'I4', 'AZ', 'VO', '.3', 'F7', '2A', 'SJ', '5Y', '-H', 'D4', '8W', 'GP', 'BU', '-Y', 'JG', 'IB', 'NY', 'G-', 'VF',
        '.M', 'RF', 'O/', '-G', '-B', 'LW', 'R6', 'C/', 'L-', 'VG', '5F', '-8', 'O-', 'FM', 'TD', 'Y5', '.9', 'KO', '-.', 'Y2', 'CY', '/G', 'NM', 'TI', 'YF', 'G6', 'I7', 'N5', 'CJ', 'XE', 'V1', 'LM',
        'DY', ',Y', 'S6', ':3', 'GZ', 'BR', 'UU', '1C', 'WU', 'OI', '3|', 'Y.', '.4', 'F/', 'NT', 'M4', 'N9', '0S', '-E', 'DW', 'Z1', 'VB', 'FW', 'G9', 'VC', 'JO', '1V', 'A+', 'JS', 'DH', '2E', 'TS',
        'S-', 'NI', '.J', 'HC', '2B', 'P2', 'FS', '9A', '7a', '7E', 'GD', 'FD', 'L3', 'TY', '5C', '7D', '6E', '6D', '2/', 'DK', '.P', '-7', 'HI', 'X1', 'U2', '1L', '4A', 'KN', '0A', 'TK', 'NQ', 'IO',
        '3E', 'Y7', 'IQ', '1S', 'WI', 'G8', '-S', '.C', 'PQ', 'N^', 'BA', 'V3', 'Z3', 'LB', '2C', '9H', 'D6', '3L', 'L1', 'SU', 'BK', 'U1', 'AO', '5a', 'SF', 'AJ', '8|', '8U', 'M7', 'LR', 'P8', 'HS',
        'DX', 'GV', '1N', 'BP', 'o1', 'DA', '.S', 'F0', 'ZB', 'FG', 'I2', 'CB', 'KK', '-9', '-6', '1E', 'DD', 'Bo', 'AW', 'VH', 'MU', 'WB', 'YS', 'ME', 'D-', 'F6', 'D5', '6C', 'E7', '-L', 'EQ', 'FC', 'voi']
#%%
from thefuzz import fuzz

def get_matches_fuzz(text):
    """
    Esta función encuentra la mejor coincidencia en similitud utilizando fuzzywuzzy.

    Elaborado por: Juan Carlos Tovar

    Args:
        palabra (str): texto.

    Returns:
        palabra (str): tupla que contiene la puntuación máxima de similitud de ratio y del token_set_ratio .
    """
        
    max_ratio = 0
    #best_match = None
    max_token_set_ratio = 0

    for item in list_asunto_no_sufiente:
        ratio = fuzz.ratio(text, item)
        token_set_ratio = fuzz.token_set_ratio(text, item)
        
        if ratio > max_ratio:

            max_ratio = ratio
            max_token_set_ratio = token_set_ratio
            #best_match = item
    
    return max_ratio,  max_token_set_ratio


def translate_words(palabra):
    """
    Esta función ajusta las palabras de un texto reemplazando tildes.

    Elaborado por: Jhosua Torres - Juan Carlos Tovar

    Args:
        palabra (str): texto.

    Returns:
        palabra (str): texto ajustado.
    """

    #Convierte variable a str (si es necesario) y pasa a MAYUS
    palabra=palabra.str.upper()  
    #Reemplaza vocales "ÁÉÍÓÚ" por "AEIOU"
    palabra=palabra.str.translate(str.maketrans("ÁÉÍÓÚ", "AEIOU")) 
    #Reemplaza vocales "ÀÈÌÒÙ" por "AEIOU"
    palabra=palabra.str.translate(str.maketrans("ÀÈÌÒÙ", "AEIOU"))  

    palabra = palabra.str.replace('¡','I')

    return palabra

def preproc(palabra): 
    """
    Esta función preprocesa una Texto en lemmas.

    Elaborado por: Juan Carlos Tovar

    Args:
        palabra (str): Texto.

    Returns:
        palabra (str): Texto preprocesado.
    """    
    custom_not_stopwords = {"estado", "cuenta", "bien", "embargo", "estados", "haber"}
    nlp.Defaults.stop_words -= custom_not_stopwords
    #Obtener objeto nlp de spacy para lemmatizar
    palabra = nlp(str(palabra))
    #Tokenizar la oracion y elimina palabras que coincidan con stopwords
    palabra = ' '.join([t.lemma_.upper() for t in palabra if not t.is_punct | t.is_stop])
    #Eliminar caracteres no ascii
    palabra = unidecode(palabra)
    return palabra

def remove_accents(input_str):
    """
    Esta función elimina acentos de un texto.

    Elaborado por: Jhosua Torres - Juan Carlos Tovar

    Args:
        input_str (str): Texto.

    Returns:

        str: Texto sin acentos.
    """
    return re.sub(r'[áéíóúÁÉÍÓÚ]', '', input_str)

def preprocess_text(text: str, remove_stopwords: bool, custom_stopwords: list = []) -> str:
    """
    Esta función preprocesa un Texto removiendo links, caracteres especiales, números, stopwords y lematizando.

    Elaborado por: Jhosua Torres - Juan Carlos Tovar

    Args:
        text (str): Texto.
        remove_stopwords (bool): Bandera para remover stopwords.
        custom_stopwords (list): Lista de stopwords personalizadas.

    Returns:
        text (str): Texto preprocesado.
    """
    # remove links
    text = unidecode(text)
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        stopwords_list = stopwords.words("spanish") + custom_stopwords

        values_to_remove = ["estado", "cuenta", "bien", "embargo", "estados", "haber"]
        stopwords_list = [value for value in stopwords_list if value not in values_to_remove]

        tokens = [w for w in tokens if not w.lower() in list(map(str.lower, stopwords_list))]
        # 3. join back together
        text = " ".join(tokens)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    # remove stopwords
    if remove_stopwords:
        tokens = [w for w in tokens if not w.lower() in list(map(str.lower, stopwords_list)) and len(w)>1]
    # return text in lower case and stripped of whitespaces
    text = " ".join(tokens).lower().strip()
    return text

def get_len_words(text):
    """
    Esta función verifica la longitud de palabras en un texto dado.

    Elaborado por: Jhosua Torres - Juan Carlos Tovar
    
    Args:
        text (str): Texto.

    Returns:
        int: Devuelve 0 si el texto contiene más de una palabra, y 1 si el texto contiene 1 sola palabra o menos.
    """
    words = text.split()
    if len(words) > 1:
        return 0
    else:
        return 1

def export_json_entities(entities_content, path_name: str):
    """
    Esta funcion exporta las entidades y metadatos en formato json.

    Elaborado por: Jhosua Torres - Juan Carlos Tovar

    Args:
        dict_entities (dict): diccionario con las 5 entidades y metadatos, si son encontradas en el documento
        path_name (str): ruta de salida del archivo json
    
    Returns:
        json_file (json): archivo json con las 5 entidades y metadatos, si son encontradas en el documento
    """
    
    with open(path_name, 'w', encoding='utf-8') as fp:
        json.dump(entities_content, fp, indent=4, ensure_ascii=False)

def categorizar_texto(texto):
    """
    Esta funcion realiza la comparacion de palabras claves con el asunto para obtener la dirección general asociada

    Elaborado por: Jhosua Torres

    Args:
        text (str): Texto.

    Returns:
        Retorna el valor otros en caso no encuentre asociada ninguna palabra clave caso contrario retorna la DG correspondiente
    """
    texto_minusculas = texto.lower()
    # Buscar palabras clave de "DGPP"
    palabras_clave_dgpp = ["presupuesto", "presupuestal", "calidad", "gasto", "tematico", "articulacion"]
    if any(palabra in texto_minusculas for palabra in palabras_clave_dgpp):
        return "DIRECCIÓN GENERAL DE PRESUPUESTO PÚBLICO"

      # Buscar palabras clave de "DGCP"
    palabras_clave_dgcp = ["contabilidad publica", "gobierno regionales", "gobiernos locales","contabili",
                        "empresas publicas", "consolidacion contable", "finanzas publicas","contaduria"]
    if any(palabra in texto_minusculas for palabra in palabras_clave_dgcp):
        return "DIRECCIÓN GENERAL DE CONTABILIDAD PÚBLICA"

      # Buscar palabras clave de "DGTP"
    palabras_clave_dgpt = ["tesoro", "creditos", "financiera", "deuda", "fiscales", "riesgos", "financieras"]
    if any(palabra in texto_minusculas for palabra in palabras_clave_dgpt):
        return "DIRECCIÓN GENERAL DEL TESORO PÚBLICO"

  # Buscar palabras clave de "DGGFRH"
    palabras_clave_dggfrhh = ["recursos", "humanos", "personal", "activo", "pensiones", "informacion", "registro"]
    if any(palabra in texto_minusculas for palabra in palabras_clave_dggfrhh):
        return "DIRECCIÓN GENERAL DE GESTIÓN FISCAL DE LOS RECURSOS HUMANOS"

  # Buscar palabras clave de "DGA"
    palabras_clave_abastecimiento = ["abastecimiento", "integrado", "adquisiciones", "muebles", "inmuebles", "innovacion"]
    if any(palabra in texto_minusculas for palabra in palabras_clave_abastecimiento):
        return "DIRECCIÓN GENERAL DE ABASTECIMIENTO"
    # Buscar palabras clave de "DGPMI"
    palabras_clave_multi = ["multianual", "politica", "evaluacion"]
    if any(palabra in texto_minusculas for palabra in palabras_clave_multi):
        return "DIRECCIÓN GENERAL DE PROGRAMACIÓN MULTIANUAL DE INVERSIONES"

  # Si no se encuentra ninguna palabra clave, devolver "OTROS"
    return "OTROS"

def replace_cl_text(texto):
    """
    Esta funcion reemplaza palabras mal escritas o 

    Elaborado por: Juan Carlos 

    Args:
        text (str): Texto.

    Returns:
        pandas.DataFrame: El DataFrame modificado con las palabras reemplazadas en la columna 'cl_text'
    """
    texto = texto.replace('remi tú', 'remite')
    texto = texto.replace('tás', 'tasa')
    texto = texto.replace('bás', 'base')
    texto = texto.replace('més', 'mesa')
    texto = texto.replace('oficín', 'oficina')
    texto = texto.replace('brén', 'brena')
    texto = texto.replace('persón', 'persona')
    texto = texto.replace('lún', 'lunes')
    texto = texto.replace('maquín', 'maquina')
    texto = texto.replace('designad tú', 'designado')
    texto = texto.replace('d tú', 'do')
    texto = texto.replace('carg tú', 'cargo')
    texto = texto.replace('remmi tú', 'remite')
    texto = texto.replace('pasajer tú', 'pasajero')
    texto = texto.replace('tribut tú', 'tributo')
    texto = texto.replace('recabadir tú', 'recabado')
    texto = texto.replace('honorar tú', 'honorario')
    texto = texto.replace('plaz tú ', 'plazo')
    texto = texto.replace('rei tú', 'remite')
    texto = texto.replace('novecient tú', 'novecientos')
    texto = texto.replace('ingenier tú', 'ingeniero')

    return texto

def correct_saludo_f1(indices, texto):
    """
    Identifica y corrige saludos en un texto basado en índices dados.

    Elaborado por: Juan Carlos 

    Args:
        text (str): Texto.

    Returns:
        list: Una lista de índices corregidos, excluyendo saludos repetidos basados en una palabra específica ('consideracion').
    """
        
    texto = texto.lower()
    list_s1 = []
    list1 = indices.copy()

    indices_consideracion = [i for i in indices if i - 13 >= 0 and texto[i-13:i] == 'consideracion']
    #print(len(indices_consideracion))

    if len(indices_consideracion) > 1:
        # for i in indices:
        #     #print(i, texto[i-13:i])
        #     if i - 13 >= 0 and texto[i-13:i] == 'consideracion':

        #         list_s1.append(i)
        #         if c is None or i < c:
        #             c = i      
        # list_s1.remove(c)
        # #print(c, list_s1)
        # list1 = [value for value in list1 if value not in list_s1]
        # #print(c, list1)
        c = min(indices_consideracion)
        list_s1 = [i for i in indices_consideracion if i != c]
        list1 = [i for i in list1 if i not in list_s1]

    return list1

def correct_despedida(indices_despedida, texto):
    """
    Identifica y corrige las despedidas en un texto basado en una longitud límite.

    Elaborado por: Jhosua Torres

    Args:
        indices_despedida (list): Una lista de índices que marcan las ubicaciones de las despedidas en el texto.
        texto (str): El texto en el que se buscan las despedidas.

    Returns:
        list: Una lista de índices corregidos, excluyendo despedidas que estén dentro de una porción específica del texto (dependiendo de su longitud).
    """

    limit=280
    len_t = len(texto)
    l=0

    listd1 = indices_despedida.copy()
    
    if len_t<=limit:
        l=0.67
    else:
        l=0.75

    t_len_t = int(len_t*l)

    nueva_lista = [i for i in listd1 if i >= t_len_t]

    return nueva_lista

def correct_saludo_f2(indices, texto):
    """
    Identifica y corrige saludos en un texto basado en una longitud límite.

    Elaborado por: Jhosua Torres

    Args:
        indices (list): Una lista de índices que marcan las ubicaciones de los saludos en el texto.
        texto (str): El texto en el que se buscan los saludos.

    Returns:
        list: Una lista de índices corregidos, excluyendo saludos que estén más allá de una porción específica del texto (dependiendo de su longitud).
    """

    limit=280
    len_t = len(texto)
    l=0
    listd1 = indices.copy()

    if len_t<=limit:
        l=0.3
    else:
        l=0.2
    
    t_len_t = int(len_t*l)+1

    nueva_lista = [i for i in listd1 if i <= t_len_t]

    return nueva_lista

def get_better_corpus(texto):
    """
    Mejora la selección de un fragmento de texto relevante de un corpus más grande, identificando los saludos y despedidas en la correspondencia formal.

    Elaborado por: Juan Carlos Tovar

    Args:
        texto (str): El texto del que se extraerá el fragmento relevante.

    Returns:
        str: El fragmento de texto relevante, excluyendo saludos y despedidas típicamente encontrados en la correspondencia formal.
    """
    
    max_ini = -1
    min_fin = len(texto)

    palabras_saludo = r'(Estimado[s]? señor[es]?|saludo cordial|(?:tengo\s+el\s+agrado\s+de\s+dirigirme|agrado\s+de\s+dirigirme\s+a\s+usted|dirigirme\s+a\s+ud|dirigirme\s+a\s+ud\s*\.|el\s+agrado\s+de\s+dirigirme\s+usted|tengo\s+el\s+agrado\s+dirigirme|tengo\s+agrado\s+dirigirme\s+usted|dirigirme\s+a\s+ud\s*,|tengo\s+el\s+agrado\s+dirigirme\s+usted|tengo\s+agrado\s+dirigirme\s+usted\s*;|el\s+agrado\s+dirigirme\s+a\s+usted|tengo\s+el\s+agrado\s+dirigirme\s+a|tengo\s+el\s+grado\s+de\s+dirigirme|dirigirme\s+a\s+ud\s*\.|dirigirme\s+o\s+usted|dirigirme\s+o\s+usted\s*,|tengo\s+elagrado\s+de\s+dirigirme|tengo\s+elagrado\s+de\s+dirigirme\s+a|elagrado\s+de\s+dirigirme\s+a\s+usted|dirigirme\s+ud)|de\s+mi(?:\s+(?:mayor|y|especial))*\s+consideracion|al mismo tiempo manifestarle|expresarle el saludo|Por medio de la presente|me dirijo a usted|me dirijo a ud|Tengo\s+el\s+agrado\s+de\s+dirigirme\s+a\s+(?:usted|Ud)|cordial saludo|dirigirme(?:\s+a)?\s+usted|saludar(?:le|lo|la)\s+cordialm(?:en|nen)te|dirijo usted|muy cordialmente)'
    palabras_despedida = r'(Atentamente|Saludos cordiales|cordiales saludos|Gracias por su atencion|En espera de una favorable respuesta|Esperando su pronta respuesta|agradeciendole|agradeciendo|(?:sin\s+otro\s+(?:en\s+)?particular|otro\s+(?:en\s+)?particular)|propici[ao]\s+la?\s+ocasion?|nos despedimos|me despido|Agradeciendo la atencion|Esperando su atencion|Agradeciendo anticipadamente|propicia la oportunidad)'

    indices_saludo = [m.end() for m in re.finditer(palabras_saludo, texto, re.IGNORECASE)]
    indices_saludo = correct_saludo_f1(indices_saludo, texto)
    indices_saludo = correct_saludo_f2(indices_saludo, texto)
    if len(indices_saludo)>0:
        max_ini = max(indices_saludo)
    #print("Índices de palabras de saludo:", indices_saludo, max_ini)

    indices_despedida = [m.start() for m in re.finditer(palabras_despedida, texto, re.IGNORECASE)]
    indices_despedida = correct_despedida(indices_despedida, texto)
    if len(indices_despedida)>0:
        min_fin = min(indices_despedida)
    #print("Índices de palabras de despedida:", indices_despedida, min_fin)

    #print("texto:", texto[max_ini+1:min_fin].strip())
    return texto[max_ini+1:min_fin].strip()
#%%
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
def get_pdf_to_tmp_local(lst_p, df):
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
        remote_filepath=df[0]
        
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

# %%
def preproc_expediente(df):
    """
    Esta funcion realiza el preprocesamiento del campo asunto aplicando diversas funciones de procesamiento

    Elaborado por: Jhosua Torres - Juan Carlos Tovar

    Args:
        df: DataFrame

    Returns:
        Retorna un json el cual contiene las entidades extraidas y asunto pre procesado
    """

    logging.basicConfig(filename=os.path.join(path_log,'doc_entity.log'), level=logging.ERROR)

    try:  
        df['TEXTO'] = translate_words(df['asunto'])
        df['t2v'] = df['TEXTO'].apply(preproc)
        for i, row in df.iterrows():
            df.at[i, 'cl_text'] = preprocess_text(row['t2v'], remove_stopwords=True, custom_stopwords=custom_stopwords)
        
        df['cl_text'] =  df['cl_text'].apply(replace_cl_text)
        df['cl_text'] =  df['cl_text'].apply(remove_accents)

        df['flag1'] = df['cl_text'].apply(get_len_words)
        
        matches = df['cl_text'].apply(get_matches_fuzz)
        df[['ratio', 'token_set_ratio']] = pd.DataFrame(matches.tolist(), index=df.index)
        df['flag2'] = np.where((df['ratio']>80)&(df['token_set_ratio']>90), 1, 0)
        
        df['dato_suficiente'] = np.where((df['flag1']==1) | (df['flag2']==1),'0','1')

        cols = ['path_file', 'nro_hoja_ruta', 'remitente', 'cl_text', 'dato_suficiente']
        df = df[cols]
        df.columns = df.columns.str.lower()

        if df['dato_suficiente'][0] == '0':
            con_ftp = get_ftp_url(var)
            lst_p = get_ftp_params(con_ftp)
            path_remote_pdf = df['path_file'][0]
            result_msg_download = get_pdf_to_tmp_local(lst_p, df['path_file'])
            path_tmp = os.path.join(path_project,'data/tmp')
            name_pdf_file = df['path_file'][0].split('/')[-1]
            path_tmp_file = os.path.join(path_tmp, name_pdf_file)
            df['path_file'] = path_tmp_file
            dict_prep = get_doc_entities(df)
            result_msg_delete = delete_pdf_tmp_file(path_tmp_file)
            df['path_file'] = path_remote_pdf
            df_dict_prep = pd.DataFrame(dict_prep, index=[0])
            
            df_dict_prep.columns = df_dict_prep.columns.str.lower()
            df_dict_prep['texto'] =  translate_words(df_dict_prep['cuerpo_ocr'])
            df_dict_prep['b_cuerpo'] = df_dict_prep['texto'].apply(get_better_corpus)
            df_dict_prep['nro_palabras_bc'] = df_dict_prep['b_cuerpo'].apply(lambda x: len(x.split()))
            df_dict_prep['t2v'] = df_dict_prep['b_cuerpo'].apply(preproc)

            for i, row in df_dict_prep.iterrows():
                df_dict_prep.at[i, 'cl_text_ocr'] = preprocess_text(row['t2v'], remove_stopwords=True, custom_stopwords=custom_stopwords)
            
            df_dict_prep['cl_text_ocr'] =  df_dict_prep['cl_text_ocr'].apply(replace_cl_text)
            df_dict_prep['cl_text_ocr'] =  df_dict_prep['cl_text_ocr'].apply(remove_accents)

            numeric_columns = df_dict_prep.select_dtypes(include=['int', 'float']).columns
            df_dict_prep[numeric_columns] = df_dict_prep[numeric_columns].astype(str)

            columns_to_drop = ['texto', 'b_cuerpo','nro_palabras_bc', 't2v']
            df_dict_prep.drop(columns=columns_to_drop, inplace=True)

            df = df.merge(df_dict_prep, how='left', on='nro_hoja_ruta')
            df['destinatario_ocr']=df['destinatario_ocr'].astype(str).apply(lambda x: unidecode(x))
            df["destinatario_ocr_siglas"]= df["destinatario_ocr"].apply(categorizar_texto)

            df = df.merge(df_direcciones_mef,on="destinatario_ocr_siglas", how="left")

        if sys.platform.startswith('win'):
            json_file = path_entity+'\\'+ df.iloc[i, 1]+'.json'
        else:
            json_file = path_entity+'/'+ df.iloc[i, 1]+'.json'

        df_col_set = set(df.columns)    
        cols = ['nro_hoja_ruta', 'path_file', 'remitente', 'cl_text', 'dato_suficiente', 'tipo_doc', 'confianza_media_ocr', 'clase_doc', 'nro_paginas', 'tiempo_extract','destinatario_ocr', 'asunto_ocr', 'cuerpo_ocr', 'remitente_ocr', 'referencia_ocr','fecha_extract','cl_text_ocr', 'destinatario_ocr_siglas', 'destinatario_ocr_id']
        cols_set = set(cols)

        missing_cols = cols_set - df_col_set

        for col in missing_cols:
            df[col] = ''

        df_dict = df.to_dict(orient='records')[0]
        export_json_entities(df_dict, json_file)
        cols = ['nro_hoja_ruta', 'cl_text', 'cl_text_ocr', 'confianza_media_ocr', 'dato_suficiente', 'destinatario_ocr_siglas', 'destinatario_ocr_id']

        return df[cols]
    except Exception as e:  
        df['error'] = str(e)[:180] + '...'
        name_file = df.iloc[0, 1] 
        traceback_str = traceback.format_exc()
        logging.error(str(datetime.now()) + f" - Error in preproc NRO_HOJA_RUTA: {name_file}; Error: {str(e)}\n{traceback_str}")
        print("Path error: ",path_error)
        df.to_json(os.path.join(path_error, name_file + "_error.json"), orient='records')
        #return df