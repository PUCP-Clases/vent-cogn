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