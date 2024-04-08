"""
En este script se encuentran las clases DOCUMENT, DOCUMENT_SCANNED_PARSER, DOCUMENT_DIGITAL_PARSER y DOCUMENT_ENTITIES.
"""
#%%Imports
import os
import traceback
import shutil
from pdf2image import convert_from_path
import numpy as np
import easyocr
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams
#from tika import parser
import os
from pdfminer.layout import LTTextContainer, LTTextLineHorizontal
import matplotlib.pyplot as plt
import re
import datetime
import json
import logging
import PyPDF2
import pandas as pd
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1
from thefuzz import fuzz, process
import cv2

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

reader = easyocr.Reader(['es','en'], gpu=False)
PATTERN = re.compile(r'[a-zA-Z0-9]')

path_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

var_path = os.path.join(path_project, 'variables.json')

var = get_json_file(var_path)

path_poppler = os.path.join(path_project, var['poppler_path'])
#print("Poppler PATH:", path_poppler)

#%%Classes
class DOCUMENT:
    def __init__(self, pdf_path: str):#, status='', bboxes='', page_start=0, text=''):
        """
        La clase DOCUMENT se encarga de validar si el documento es digital o escaneado,
        ademas de extraer el tipo de documento (carta, oficio, memorando, etc) y el numero de paginas.
        """
        self.pdf_path = pdf_path
        self.digital = ''
        self.status=''
        self.pdf_type()
        #self.bboxes=bboxes
        #self.page_start=page_start
        #self.text=text
        #self.digital=digital
        #self.doc_type = ''
        self.num_pages = 0

        #self.get_doc_type()
        self.num_pages = self.count_pdf_pages()

    def export_json_entities(self, entities_content, path_name: str):
        """
        Esta funcion exporta las entidades y metadatos en formato json.

        Elaborado por: Juan Carlos Tovar

        Args:
            dict_entities (dict): diccionario con las 5 entidades y metadatos, si son encontradas en el documento
            path_name (str): ruta de salida del archivo json
        
        Returns:
            json_file (json): archivo json con las 5 entidades y metadatos, si son encontradas en el documento
        """
        
        with open(path_name, 'w', encoding='utf-8') as fp:
            json.dump(entities_content, fp, indent=4, ensure_ascii=False)
        
    def pdf_type(self):
        """
        Esta funcion valida si el documento es digital o escaneado, para esto se utiliza la función extract_text.

        Elaborado por: Jhosua Torres

        Args:
            pdf_path (str): ruta del documento
        
        Returns:
            digital (bool): True si es digital, False si es escaneado
        """
        print('Class Document: 01. DOCUMENT TYPE')
        try:
            text_ = extract_text(self.pdf_path, page_numbers=[0], laparams=LAParams())
            self.status=True
            if text_.strip():
                self.digital =  True
            else:
                self.digital =  False
        except Exception as e:
            self.status=False
            print(e)

        print('Class Document: RESULT: DIGITAL DOC', self.digital)
    
    def count_pdf_pages(self):
        """
        Esta funcion cuenta el numero de paginas del documento.

        Elaborado por: Juan Carlos Tovar

        Args:
            pdf_path (str): ruta del documento
        
        Returns:
            num_pages (int): numero de paginas del documento
        """
        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            return num_pages
    
    def get_doc_type(self, text):
        """
        Esta funcion extrae el tipo de documento (carta, oficio, memorando, etc) a partir de las palabras claves
        que se encuentran en el documento.

        Elaborado por: Juan Carlos Tovar

        Args:
            text (str): texto del documento
        
        Returns:
            doc_type (str): tipo de documento
        """
        self.doc_types = ['CARTA', 'INFORME', 'MEMORANDO', 'OFICIO']
        self.doc_added_keys = ['multiple', 'múltiple','circular']

        #print("Text in get_doc_type: ", self.text)
        doc_type = 'OTROS'
        similarity_dict = {}
        #replace 'qficio' or '0ficio' or '0fici0' by 'oficio' in self.text
        text = re.sub(r'qficio|0ficio|0fici0|oFrcro|QHCIONL|oFtcto'.upper(), 'OFICIO', text, flags=re.IGNORECASE)
        for word in self.doc_types:
            matches = process.extractBests(word, text.upper().split())
            #print(word,"-",matches)
            matches = [item for item in matches if len(item[0]) > 4]
            if matches:
                matches = max(matches, key=lambda x: x[1])
                max_match = matches[0]
                max_prob = matches[1]
                if max_prob >= 80 and len(max_match)>=5:
                    #print(word,"-",matches)
                    text = text.replace(max_match, word)
            else:
                pass     
                #similarity_dict[word] = max_prob
        #print("text: ", text)
        patt = r"(?=("+'|'.join(self.doc_types)+r"))"
        match_n1 = re.findall(patt.lower(), text.lower())
        if match_n1: 
            # print(match_n1)
            m_n1 = re.search(match_n1[0], text.lower())
            doc_type = match_n1[0].upper()
            #print('TEXTO RESPALDO: \n', self.text[m_n1.start():m_n1.start()+50].strip())
            if match_n1[0].upper() in ['MEMORANDO', 'OFICIO']:
                patt2 = r"(?=("+'|'.join(self.doc_added_keys)+r"))"
                match_n2 = re.findall(patt2.lower(), text.lower())
                if match_n2: 
                    # print(match_n2)
                    m_n2 = re.search(match_n2[0], text.lower())
                    if m_n2.start()-m_n1.end() < 10:
                        doc_type += " " + match_n2[0].upper()
        print("get_doc_type F:", doc_type.upper())
        return doc_type              
        
        
    def plot_with_bounding_boxes(self, digital, id2bbox, page_start, end_page):
        """
        Esta funcion grafica el documento con sus bounding boxes.

        Elaborado por: Juan Carlos Tovar

        Args:
            image (array): imagen del documento
            id2bbox (dict): id de posicion de bboxes
            page_start (int): pagina de inicio del documento
            end_page (int): pagina de fin del documento
        
        Returns:
            None
        """
        try:
            def reset_outer_keys(dictionary):
                """
                Esta funcion reemplaza las claves exteriores de un diccionario con números ascendentes comenzando desde 1.
                
                Elaborado por: Juan Carlos Tovar - Jhosua Torres 
   
                Args:
                dictionary (dict): El diccionario de entrada.

                Returns:
                Dict: Un nuevo diccionario con las claves exteriores reemplazadas por números ascendentes.
                """
                new_dict = {}
                new_key = 1
                for key, value in dictionary.items():
                    new_dict[new_key] = value
                    new_key += 1
                return new_dict

            id2bbox = reset_outer_keys(id2bbox)

            if not self.status:
                print("DOC is not valid.")
                return

            if digital:
                images = convert_from_path(self.pdf_path, first_page=page_start+1, last_page=end_page+1, poppler_path=path_poppler)
            else:
                images = convert_from_path(self.pdf_path, first_page=page_start+1, last_page=end_page, poppler_path=path_poppler)

            annotated_images = []
            for page_num, image in enumerate(images):#, start=page_start):
                fig, ax = plt.subplots(figsize=(8, 10))
                ax.imshow(image)
                for idx, bbox in enumerate(id2bbox[page_num+1].values()):
                    x1, y1, x2, y2 = bbox
                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b')
                    plt.text(x1, y1 - 10, str(idx), color='blue', fontsize=8)
                ax.axis('off') 
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                annotated_images.append(image)
                plt.close(fig)

            final_image = np.concatenate(annotated_images, axis=0)

            return final_image
        except Exception as e:
            print(e)
            return None 


class DOCUMENT_SCANNED_PARSER(DOCUMENT):
    def __init__(self, pdf_path: str, n_pages: int, height_ths: float = 0.8, 
                 width_ths: float = 0.5, conf: float = 0.1,
                 output_dir: str = "", save_img: bool = False):
        """
        La clases DOCUMENT_SCANNED_PARSER se encarga de extraer el texto y su ubicacion a nivel de pixeles (bboxes),
        a traves de EasyOCR, realizando los siguientes pasos:
        1. Extraer la imagen en array del pdf
        2. Extraer los textos y bounding boxes con easyOCR

        Elaborado por: Juan Carlos Tovar - Jhosua Torres

        Args:
            pdf_path (str): ruta del documento
            height_ths (float): umbral de altura para el texto
            width_ths (float): umbral de ancho para el texto
            conf (float): umbral de confianza para el texto
            output_dir (str): ruta de salida de la imagen
            save_img (bool): True si se desea guardar la imagen, False en caso contrario
        
        Returns:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
        """
        self.pdf_path = pdf_path
        self.height_ths = height_ths
        self.width_ths = width_ths
        self.conf = conf
        self.output_dir = output_dir
        self.save_img = save_img
        
        self.W_img, self.H_img = 0,0
        self.name = ''
        #self.n_pages = self.get_n_pages()
        self.n_pages = n_pages
        self.end_keyword = "atentamente"
        self.end_keywords = ['atentamente', 'saludos']#, 'sinotropart']
        self.end_keyword_exists = False
        self.end_page = 1 

        self.doc_type =''
        self.bboxes = []

        self.id2text = {}
        self.id2bbox = {}
        self.text = ''
        self.results = []
        self.status = False
        self.page_start = 0
        self.confidence_scores = np.array([])
        self.avg_confidence = 0
        #self.image_array = self.pdf_to_image()
        self.images = self.pdf_to_image()

        if self.status:
            print('*** PROCESSING ***')
            print('* Get 15 firsts bboxes *')
            self.text = self.get_text_from_15_firsts_bboxes()
            print('* Get Doc Class *')
            self.doc_type = super().get_doc_type(self.text)
            print("Doc Class: ", self.doc_type)
            if self.doc_type == 'OTROS':
                self.text = ''
                pass
            else:
                print("***get_all_text***")
                #self.get_bboxes()
                self.get_all_text()
                self.get_avg_confidence()

        #super().__init__(self.pdf_path, self.status, self.bboxes, 1, self.text)
        
        #self.doc_type = super().get_doc_type(self.text)

    def get_confidence_scores(self, ocr_results):
        """
        Esta funcion extrae la confianza de cada texto.

        Elaborado por: Jhosua Torres

        Args:
            ocr_results (list): lista de resultados de easyOCR

        Returns:
            confidence_scores (list): lista de confianza de cada texto
        """
        ocr_results = [result for result in ocr_results if result[2] > self.conf and len(result[1].strip())>3]
        confidence_scores = np.array([result[2] for result in ocr_results])
        #avg_confidence = np.mean(confidence_scores)
        return confidence_scores

    def get_blur_image(self, image_array):
        """
        Esta funcion transforma la imagen a escala de grises, aplica un filtro canny, un filtro morfologico y un filtro gaussiano.

        Elaborado por: Jhosua Torres
        
        Args:
            image_array (array): imagen del documento

        Returns:
            img_blur (array): imagen del documento con filtros
        """

        img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 50, 100, apertureSize = 3)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))

        opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        mask = np.zeros(img_gray.shape, np.uint8)
        for c in cnts:
            cv2.drawContours(mask, [c], -1, (255,255,255),2)

        img_dst = cv2.inpaint(img_gray, mask, 3, cv2.INPAINT_TELEA)

        img_blur = cv2.GaussianBlur(img_dst, (3,3), 0)

        return img_blur

    def get_avg_confidence(self):

        """
        Esta funcion obtiene la confianza promedio.
        
        Elaborado por: Juan Carlos Tovar

        """
        self.avg_confidence = np.round(np.mean(self.confidence_scores), 3)

    def get_text_from_15_firsts_bboxes(self):
        """
        Esta funcion extrae los textos y bounding boxes con easyOCR de los 15 primeros bboxes.

        Elaborado por: Juan Carlos Tovar

        Args:
            image_array (array): imagen del documento

        Returns:
            text (str): texto del documento de los 15 primeros bboxes
        """
        image_array = np.array(self.images[0])
        img_blur = self.get_blur_image(image_array)
        self.results = reader.readtext(img_blur, height_ths=self.height_ths, width_ths=self.width_ths)
        text = ''
        for i in self.results[:30]:
            text = text +" "+ i[1]

        return text


    def pdf_to_image(self):
        """
        Esta funcion extrae la imagen en array del pdf.

        Elaborado por: Juan Carlos Tovar

        Args:
            pdf_path (str): ruta del documento
        
        Returns:
            image_array (array): imagen del documento
        """

        print('01. IMAGE TO PDF')
        try:
            images = convert_from_path(self.pdf_path, poppler_path=path_poppler)#, first_page=1, last_page=1)
            self.status = True
        except Exception as e:
            print(e)

        name = self.pdf_path.split('/')[-1].lower()
        self.name = name.split('.pdf')[0]

        self.W_img, self.H_img = images[0].size

        if self.save_img:
            images[0].save(os.path.join(self.output_dir, self.name + '_0.jpg'), 'JPEG')

        return images #np.array(images[0])

    def get_n_pages(self):
        n_pages = 0
        """0
        Esta funcion cuenta el numero de paginas del documento.

        Elaborado por: Jhosua Torres

        Args:
            pdf_path (str): ruta del documento

        Returns:
            n_pages (int): numero de paginas del documento
        """
        file = open(self.pdf_path, 'rb')
        parser = PDFParser(file)
        document = PDFDocument(parser)
        try:
            n_pages = resolve1(document.catalog['Pages'])['Count']
        except:
            n_pages = len(list(extract_pages(self.pdf_path)))
        return n_pages

    def get_all_text(self):
        """
        Esta funcion extrae los textos y bounding boxes con easyOCR.
        
        Elaborado por: Juan Carlos Tovar

        Args:
            image_array (array): imagen del documento

        Returns:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
        """
        print('02. LOOP ALL PAGES')
        page_counter = 0
        for image in self.images:
            image_array = np.array(image)
            #print(image_array.shape)
            id2bbox, id2text, confidence_scores = self.get_bboxes(image_array=image_array)
            self.confidence_scores = np.concatenate((self.confidence_scores, confidence_scores))
            self.id2text[page_counter+1] = id2text
            self.id2bbox[page_counter+1] = id2bbox
            #print("id2text: ", id2text)
            page_counter +=1
            if self.n_pages == 1 or self.end_keyword_exists:
                self.end_page = page_counter
                break 

    
    def get_bboxes(self, image_array):
        """
        Esta funcion extrae los textos, bounding boxes y confianza con easyOCR.

        Elaborado por: Jhosua Torres

        Args:
            image_array (array): imagen del documento
        
        Returns:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
            confidence_scores (list): lista de confianza de cada texto
        """

        print('03. BBOXS AND TEXT EXTRACTION')
        id2text = {}
        id2bbox = {}
        img_blur = self.get_blur_image(image_array)
        results = reader.readtext(img_blur, height_ths=self.height_ths, width_ths=self.width_ths)

        confidence_scores = self.get_confidence_scores(results)

        i = 0 
        for result in results:
            if result[2] > self.conf and len(result[1].strip())>3:
                x1, y1 = result[0][0]
                x2, y2 = result[0][2]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                id2bbox[i] = x1, y1, x2, y2
                id2text[i] = result[1].strip()
                self.bboxes.append((i, x1,y1,x2,y2))
                found_items = [item for item in self.end_keywords if item in result[1].strip().lower()]
                #print("Found items: ", found_items)
                if found_items or fuzz.partial_ratio(self.end_keyword, result[1].strip().lower())>90:
                #if self.end_keyword in result[1].strip().lower() or fuzz.partial_ratio(self.end_keyword, result[1].strip().lower()):
                    self.end_keyword_exists = True 
                i+=1
        #self.text = "\n".join(id2text.values())

        return id2bbox, id2text, confidence_scores
                

class DOCUMENT_DIGITAL_PARSER(DOCUMENT):
    def __init__(self, pdf_path: str, n_pages: int, limit_y: int = 10):
        """
        Ls clase DOCUMENT_DIGITAL_PARSER se encarga de extraer el texto y su ubicacion a nivel de pixeles (bboxes),
        para esto valida primero si este contiene texto y su metadata a traves de PDFMINER SIX.
        El procedimiento sigue los siguientes pasos solo si el atributo status es valido:
        1. Extraer tamaños de pdf digital y en su formato imagen
        2. Extraer los textos y bounding boxes de los contenedores de texto en linea horizontal
        3. Ordenar los bboxes de arriba hacia abajo y derecha a izquierda

        Elaborado por: Juan Carlos Tovar

        Args:
            pdf_path (str): ruta del documento
            limit_y (int): umbral de distancia vertical para considerar que el texto pertenece a una linea

        Returns:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
        """

        self.pdf_path = pdf_path
        self.limit_y = limit_y
        self.H_pdf, self.W_pdf = 0, 0
        self.H_img, self.W_img = 0, 0
        self.bboxes = []
        self.id2text = {}
        self.id2bbox = {}

        #self.n_pages = self.get_n_pages()
        self.n_pages = n_pages
        self.end_keyword = "atentamente"
        self.end_keywords = ['atentamente', 'saludos']
        self.end_keyword_exists = False
        self.end_page = 1

        self.name = ''
        self.doc_type =''
        self.save_img = False
        self.status = ''
        self.page_start = 0
        self.text = ''
        self.check_pdf()
        # print(self.status)

        if self.status:
            print('*** PROCESSING ***')
            print('* GET PAGE START *')
            self.get_page_start()
            print("Page Start: ", self.page_start)
            self.doc_type = super().get_doc_type(self.text)
            print("Doc Class: ", self.doc_type)
            
            if self.doc_type == 'OTROS':
                self.text = ''
                pass
            else:
                print('* GET SIZE AND CONVERT *')
                self.get_size_and_convert()
                print("Page Size: ", self.W_img, self.H_img)

                print('* GET BBOXES *')
                self.get_all_text()
                print("Bboxs: ",self.bboxes)
                
                #print('* SORT BBOXES *')
                #self.sort_bboxes()
                #print("Bboxs Sorted")

    def check_pdf(self):
        """
        Esta funcion valida si el documento es digital, para esto se utiliza la función extract_text.

        Elaborado por: Jhosua Torres

        Args:
            pdf_path (str): ruta del documento
        
        Returns:
            digital (bool): True si es digital, False si es escaneado
        """
        print('01. DOCUMENT VALIDATION')
        try:
            text = extract_text(self.pdf_path, page_numbers=[0], laparams=LAParams())
            if text.strip():
                self.status =  True
            else:
                self.status =  False
        except Exception as e:
            print(e)
            self.status =  False

        print('RESULT: ', self.status)
    
    def get_n_pages(self):
        """
        Esta funcion cuenta el numero de paginas del documento.

        Elaborado por: Jhosua Torres

        Args:
            pdf_path (str): ruta del documento

        Returns:
            n_pages (int): numero de paginas del documento
        """
        n_pages = 0
        print("Get N Pages:")
        file = open(self.pdf_path, 'rb')
        parser = PDFParser(file)
        document = PDFDocument(parser)
        try:
            n_pages = resolve1(document.catalog['Pages'])['Count']
        except:
            n_pages = len(list(extract_pages(self.pdf_path))) 

        print("n_pages: ",  n_pages)
        return n_pages
        
    
    def get_page_start(self):
        """
        Esta funcion extrae la pagina de inicio del documento, para esto se utiliza la función extract_pages.

        Elaborado por: Juan Carlos Tovar

        Args:
            pdf_path (str): ruta del documento
        
        Returns:
            page_start (int): pagina de inicio del documento
        """
        def replace_and_add_suffix(input_string):
            replacement_dict = {
                "ofrcro": "oficio",
                "oftcto": "oficio",
                "qhcionl":"oficio",
                "qeicio_n":"oficio",
                "qeicio":"oficio",
                "mútttple":"múltiple",
                # Add more replacements as needed
                    }
            for old_value, new_value in replacement_dict.items():
                input_string = input_string.replace(old_value, new_value)
            
            return  input_string
        
        j=0
        class MatchFoundException(Exception):
            pass

        page_layout = extract_pages(self.pdf_path)
        pattern = re.compile(r'(memora|oficio|carta|informe)')

        try:
            for page in page_layout:
                print('Page N° ',j)
                for element in page:  
                    flag = False
                    if isinstance(element, LTTextContainer):
                        if element.get_text().strip():
                            for text_line in element:
                                if isinstance(text_line, LTTextLineHorizontal):
                                    text_lines=text_line.get_text().strip().lower()
                                    match = re.search(pattern,replace_and_add_suffix(text_lines))
                                    if match:
                                        print(replace_and_add_suffix(text_lines))
                                        flag = True
                                        print("MatchFoundException N1")
                                        self.page_start = j
                                        self.text = extract_text(self.pdf_path, page_numbers=[j]).strip()
                                        raise MatchFoundException
                                    else:
                                        print("Match NOT FOUND")
                            if flag:
                                print("MatchFoundException N2")
                                
                                raise MatchFoundException
                j+=1 
        except MatchFoundException:
            print("MatchFoundException N3")
            pass

    def get_size_and_convert(self):
        """
        Esta funcion extrae los tamaños de pdf digital y en su formato imagen.

        Elaborado por: Juan Carlos Tovar

        Args:
            pdf_path (str): ruta del documento
            page_start (int): pagina de inicio del documento

        Returns:
            W_img (int): ancho de la imagen
            H_img (int): alto de la imagen
        """
        print("get_size_and_convert: Page N°", self.page_start)
        images = convert_from_path(self.pdf_path, first_page=self.page_start+1, last_page=self.page_start+1, poppler_path=path_poppler)

        name = self.pdf_path.split('/')[-1].lower()
        self.name = name.split('.pdf')[0]

        self.W_img, self.H_img = images[0].size

        print('01. GET DATA FROM DOC & IMG')

        if self.save_img:
            images[0].save(os.path.join('test',self.name + '_0.jpg'), 'JPEG')

    def get_all_text(self):
        """
        Esta funcion extrae los textos y bounding boxes de los contenedores de texto en linea horizontal.

        Elaborado por: Jhosua Torres

        Args:
            pdf_path (str): ruta del documento
            page_start (int): pagina de inicio del documento
            W_img (int): ancho de la imagen
            H_img (int): alto de la imagen

        Returns:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
        """

        #page_counter = 0
        page_counter = self.page_start-1

        range_pages = [i for i in range(self.page_start, self.n_pages)]
        for page_layout in extract_pages(self.pdf_path, page_numbers=range_pages):
            print("Page Counter: ", page_counter)
            bboxes, id2text = self.get_bboxes(page_layout=page_layout)
            id2bbox, id2text = self.sort_bboxes(bounding_boxes=bboxes, id2text=id2text)
            self.id2text[page_counter+1] = id2text
            self.id2bbox[page_counter+1] = id2bbox
            page_counter +=1
            # print(self.end_keyword_exists)
            if self.n_pages == 1 or self.end_keyword_exists:
                self.end_page = page_counter
                break


    def get_bboxes(self, page_layout):
        """
        Esta funcion extrae los textos y bounding boxes de los contenedores de texto en linea horizontal.
        
        Elaborado por: Jhosua Torres

        Args:
            page_layout (dict): pagina del documento

        Returns:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
        """
        print('02. GET BBOXES WITH PDFMINER')
        i = 0
        bboxes = []
        id2text = {}
        #range_pages = [i for i in range(self.page_start, self.page_start+1)]
        #page_layout = next(extract_pages(self.pdf_path, page_numbers=range_pages))
        _, _, self.W_pdf, self.H_pdf =  page_layout.bbox
        print("02.1 Page Layout: ")
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                if element.get_text().strip():
                    for text_line in element:
                        if isinstance(text_line, LTTextLineHorizontal) and bool(PATTERN.search(text_line.get_text())):
                            x1,y1,x2,y2 = text_line.bbox
                            x1,y2,x2,y1 = int(x1*self.W_img/self.W_pdf), self.H_img-int(y1*self.H_img/self.H_pdf), \
                                          int(x2*self.W_img/self.W_pdf),self.H_img-int(y2*self.H_img/self.H_pdf)
                            bboxes.append((i, x1,y1,x2,y2))
                            self.bboxes.append((i, x1,y1,x2,y2))
                            id2text[i] = text_line.get_text().strip()
                            #print("END KEY WORDS: ", self.end_keywords)
                            #print("TEXT LINE: ", text_line.get_text().strip().lower())
                            found_items = [item for item in self.end_keywords if item in text_line.get_text().strip().lower()]
                            #print("Found items: ", found_items)
                            if found_items:
                            #if self.end_keyword in text_line.get_text().strip().lower():
                                self.end_keyword_exists = True 
                            i+=1 
        return bboxes, id2text

    def sort_bboxes(self, bounding_boxes, id2text):
        """
        Esta funcion ordena los bboxes de arriba hacia abajo y derecha a izquierda.
        
        Elaborado por: Juan Carlos Tovar

        Args:
            bboxes (list): lista de bboxes

        Returns:
            bboxes (list): lista de bboxes ordenados
        """
        def area_boxes_close(box1, box2, y_tolerance=5):
            """
            Esta funcion valida si el area de dos bboxes son cercanas.

            Elaborado por: Juan Carlos Tovar

            Args:
                box1 (list): bbox 1
                box2 (list): bbox 2
                y_tolerance (int): umbral de distancia vertical para considerar que el texto pertenece a una linea
            
            Returns:
                True si el area de dos bboxes son cercanas, False en caso contrario
            """
            return abs(box1[2] - box2[2]) <= y_tolerance

        #bounding_boxes = self.bboxes # [(pos, x1,y1, x2, y2)]
        sorted_boxes = sorted(bounding_boxes, key=lambda box: (box[2], box[1]))

        final_ordered_boxes = []

        current_row = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            if area_boxes_close(box, current_row[-1]):
                current_row.append(box)
            else:
                current_row = sorted(current_row, key=lambda box: box[1])
                final_ordered_boxes.extend(current_row)
                current_row = [box]

        if current_row:
            current_row = sorted(current_row, key=lambda box: box[1])
            final_ordered_boxes.extend(current_row)

        # bboxes = final_ordered_boxes
        id2bbox = {i:x[1:] for i,x in enumerate(final_ordered_boxes)}
        id2text = {i:id2text[x[0]]for i,x in enumerate(final_ordered_boxes)}

        return id2bbox, id2text


class DOCUMENT_ENTITIES:

    def __init__(self, id2text, id2bbox, y_limit = 10):
        """
        La clase DOCUMENT_ENTITIES de extraer las 5 entidades, 4 de ellas con palabras claves y la 5ta con logica geometrica.
        Tener en cuenta que esto sera posible solo si  dentro del documento contiene las palabras claves y es similar 
        al formato de documento con el cual se construyo esta clase.

        Consideraciones:
        1. La entidad PARA debe contener cualquiera de las palabras claves encontradas en la data, ademas no acepta caracteres especiales
        2. la entidad DE se corta hasta max 2 bloques adicionales donde se encuentra la palabra clave para evitar los pies de paginas
        3. En caso de multiples paginas, los diccionarios de ID2TXT e ID2BBOX se reconstruyen para que sean un diccionario de una sola hoja grande
        4. Se anadio un filtro final para eliminar algunos caracteres extras: "."

        Elaborado por: Juan Carlos Tovar

        Args:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
            y_limit (int): umbral de distancia vertical para considerar que el texto pertenece a una linea
        
        Returns:
            entities_content (dict): diccionario con las 5 entidades, si son encontradas en el documento
        """
        # self.key_para = 'señor'
        # key_asunto = 'asunto'
        # key_referencia = 'referencia'
        # key_corpus = ''
        # key_de = 'atentamente'
        self.bag4keys = {'PARA': ['señor', 'señora', 'señorita','doctor', 'doctora','señores',"SRS"], 
                        'ASUNTO': ['asunto'], 
                        'REF': ['referencia', 'ref.', 'ref', 'referente'], 
                        'DE': ['atentamente', 'saludoscord']}
        self.keys = ['señor', 'asunto' , 'ref', 'atentamente']
        self.keys_by_regex = ['PARA', 'ASUNTO', 'REF', 'DE']
        self.entidades = ['PARA', 'ASUNTO', 'REF', 'DE', 'CUERPO']
        self.key2ent = {k: ent for k, ent in zip(self.keys_by_regex, self.entidades[:-1])}
        self.ent2key = {ent: k for k, ent in zip(self.keys_by_regex, self.entidades[:-1])}

        # self.entity_startpoint = {ent :-1 for ent in self.entidades}
        self.y_limit = y_limit
        self.id2text = id2text
        self.id2bbox = id2bbox
        #self.n_texts = max(list(self.id2bbox.keys()))+1
        self.id2text = self.transform_content(x=id2text)
        self.id2bbox = self.transform_content(x=id2bbox)

        self.id2text = self.remove_cid_text(dict=self.id2text)

        self.n_texts = len(self.id2bbox)
        #print(self.id2text, self.n_texts)

        self.entity_startpoint = self.search_entities_by_keys()
        print("search_entities_by_keys: ",self.entity_startpoint)
        self.get_corpus_startpoint()
        print("get_corpus_startpoint: ",self.entity_startpoint)

        self.entities_content = self.get_entities()

        self.filter_entities()

        #for key, value in self.entities_content.items():
        #    if isinstance(value, str):  # Check if the value is a string
        #        self.entities_content[key] = re.sub(r'\s{2,}', ' ', value)
    
    def remove_cid_text(self, dict):
        """
        Esta funcion elimina los caracteres especiales (cid:*) de los textos.

        Elaborado por: Jhosua Torres

        Args:
            dict (dict): diccionario con los textos

        Returns:
            dict (dict): diccionario sin caracteres especiales
        """
    
        for key, value in dict.items():
            dict[key] = re.sub(r'\(cid:\d+\)', '', value)

        return dict

    def transform_content(self, x):
        """
        Esta funcion transforma el diccionario de ID2TXT e ID2BBOX en un diccionario de una sola hoja grande.
        
        Elaborado por: Jhosua Torres

        Args:
            x (dict): diccionario de ID2TXT e ID2BBOX

        Returns:
            id2content (dict): diccionario de una sola hoja grande
        """
        print("Type in transform_content: ", type(x))
        id2content = []
        for content_dict in x.values(): 
            id2content+= list(content_dict.values())

        id2content = {i: v for i,v in enumerate(id2content)}
        return id2content
    
    def get_entities(self):
        """
        Esta funcion extrae las 5 entidades, 4 de ellas con palabras claves y la 5ta (CUERPO) con logica geometrica.

        Elaborado por: Juan Carlos Tovar - Jhosua Torres

        Args:
            entity_startpoint (dict): id de posicion de cada entidad
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes

        Returns:
            dict_entities (dict): diccionario con las 5 entidades, si son encontradas en el documento
        """
        entidades = ['PARA', 'ASUNTO', 'REF', 'CUERPO', 'DE']
        ent_to_remove = []

        dict_entities = {e: '' for e in entidades}

        for e in entidades:
            if self.entity_startpoint[e] == -1:
                #entidades.remove(e)
                ent_to_remove.append(e)

        entidades = [e for e in entidades if e not in ent_to_remove]

        id2text = self.id2text.copy()
        id2text = {key: value for key, value in id2text.items() if len(value) > 1}

        if len(entidades):
            for i in range(len(entidades)-1):
                x_start = self.entity_startpoint[entidades[i]] #+ 1 
                #if entidades[i] == 'CUERPO':
                #    x_start = self.entity_startpoint[entidades[i]]
                x_end = self.entity_startpoint[entidades[i+1]]
                text_content = ' '.join([id2text[x].strip() for x in range(x_start, x_end) if x in id2text.keys()])
                dict_entities[entidades[i]] = text_content

            x_start = self.entity_startpoint[entidades[-1]]# + 1 
            #if entidades[-1] == 'CUERPO':
            #    x_start = self.entity_startpoint[entidades[-1]]
            if entidades[-1]== "DE":
                next_keys = [key for key in sorted(id2text.keys()) if key > x_start][:4]
                selected_values = [id2text.get(key, '') for key in [x_start] + next_keys]
                text_content = ' '.join([self.remove_de_uncommon_words(x.strip()) for x in selected_values if x])
                #text_content = ' '.join([self.remove_de_uncommon_words(id2text[x].strip()) for x in range(x_start, min(x_start+5,self.n_texts)) if x in id2text.keys()])
            else:
                text_content = ' '.join([id2text[x].strip() for x in range(x_start, self.n_texts) if x in id2text.keys()])

            #dict_entities[entidades[-1]] = ' '.join([self.id2text[x] for x in range(x_start, self.n_texts)])
            dict_entities[entidades[-1]] = text_content 

            dict_entities['FECHA_EXTRACCION'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        #print("ENTIDADES: ", dict_entities)
        return dict_entities
    
    def remove_de_uncommon_words(self, value):
        """
        Esta funcion elimina las palabras no comunes de la entidad DE.

        Elaborado por: Juan Carlos Tovar

        Args:
            value (str): texto de la entidad DE

        Returns:
            value (str): texto de la entidad DE sin palabras no comunes
        """

        pattern = re.compile(r'^(?!(www\.|C\.C\.)).*', re.IGNORECASE)
        if pattern.match(value):
            return value
        else:
            return ''

    def filter_entities(self):
        """
        Esta funcion filtra las 5 entidades, 4 de ellas con palabras claves y la 5ta (CUERPO) con logica geometrica.

        Elaborado por: Juan Carlos Tovar

        Args:
            entity_startpoint (dict): id de posicion de cada entidad
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes

        Returns:
            dict_entities (dict): diccionario con las 5 entidades, si son encontradas en el documento
        """

        def match_words_in_list(word_list, text):
            """
            Esta funcion valida si una palabra pertenece a una lista de palabras.

            Elaborado por: Juan Carlos Tovar

            Args:
                word_list (list): lista de palabras
                text (str): texto del documento

            Returns:
                matches (list): lista de palabras que pertenecen a la lista de palabras
            """
            pattern = r'\b(?:' + '|'.join(map(re.escape, word_list)) + r')\b'
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            return matches

        #print(self.bag4keys.items())
        for entity, word_list in self.bag4keys.items():
            if self.entities_content[entity]:
                #print(entity)
                if len(word_list) > 1:
                    matches = match_words_in_list(word_list=word_list, text=self.entities_content[entity])
                    if matches:
                        # print(matches)
                        if matches[0].lower() == 'ref':
                            try:
                                content = self.entities_content[entity].split(matches[0]+".", 1)[-1].strip()
                                self.entities_content[entity] = content # .replace(":","").strip()
                            except:
                                pass
                        else:
                            self.entities_content[entity] = self.entities_content[entity].split(matches[0], 1)[-1].strip()
                else:
                    try:
                        #print("Trying match: ", word_list[0],"--", self.entities_content[entity].lower())
                        match = re.search(word_list[0], self.entities_content[entity].lower())
                        self.entities_content[entity] = self.entities_content[entity][match.end():]
                    except Exception as e:
                        print("filter_entities error: ", e)
                        continue
        self.entities_content['PARA'] = self.entities_content['PARA'].replace(":","").strip()
        self.entities_content['ASUNTO'] = self.entities_content['ASUNTO'].replace(":","").strip()
        self.entities_content['REF'] = self.entities_content['REF'].replace(":","").strip()
        self.entities_content['DE'] = self.entities_content['DE'].replace(",","").strip()

        # Remove multiple spaces in string values
        for key, value in self.entities_content.items():
            if isinstance(value, str):
                self.entities_content[key] = re.sub(r'\s+', ' ', value)

        return None
    
    def search_entities_by_keys(self):
        """
        Esta funcion extrae las 4 entidades con palabras claves.

        Elaborado por: Jhosua Torres

        Args:
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes

        Returns:
            entity_startpoint (dict): id de posicion de cada entidad
        """
        def match_words_in_list(word_list, text):
            """
            Esta funcion valida si una palabra pertenece a una lista de palabras.

            Elaborado por: Jhosua Torres

            Args:
                word_list (list): lista de palabras
                text (str): texto del documento

            Returns:
                matches (list): lista de palabras que pertenecen a la lista de palabras
            """
            #pattern = r'\b(?:' + '|'.join(map(re.escape, word_list)) + r')\b'
            if "atentamente" in word_list:
                pattern = r"(?:{})".format('|'.join(map(re.escape, word_list)))
            else:
                pattern = r"\b(?:{})\b".format('|'.join(map(re.escape, word_list)))
            text = text.replace(" ","")
            
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            return matches
        
        st = 0
        keys = self.keys_by_regex.copy() # ['PARA', 'ASUNTO', 'REF', 'DE']
        entity_startpoint = {ent :-1 for ent in self.entidades}

        while len(keys) > 0:
            #print(keys)
            if keys[0] == 'REF':
                iter_values = zip(list(self.id2text.keys())[st:-5], list(self.id2text.values())[st:-5])
            else:
                iter_values = zip(list(self.id2text.keys())[st:], list(self.id2text.values())[st:])
            bag_list = self.bag4keys[keys[0]]
            for idx, text in iter_values:
                if len(keys)==0:
                    return entity_startpoint
                
                matches = match_words_in_list(bag_list, text)
                if matches:
                    #print(bag_list, matches)
                    text = text.replace(" ","")
                    x = re.search(matches[0], text, re.IGNORECASE)
                    #print(x)
                    if x:
                        entity_startpoint[keys[0]] = idx
                        st = idx
                        break #keys.remove(keys[0]) 
            
            if len(keys)==0:
                    return entity_startpoint
            keys.remove(keys[0]) 
        #bloque: PARA == -1    
        if entity_startpoint['PARA'] == -1:
            first_8_items = list(self.id2text.items())[0:8]
            pattern = re.compile(r'CARTA|OFICIO|INFORME|MEMORA', flags=re.IGNORECASE)
            matching_items = [(key, value) for key, value in first_8_items if pattern.search(value)]
            entity_startpoint['PARA'] = matching_items[0][0] + 1
            
        return entity_startpoint
     
    def get_corpus_startpoint(self):
        #print("self.id2bbox: ", self.id2bbox)
        """
        Esta funcion extrae la entidad CUERPO con logica geometrica.

        Elaborado por: Juan Carlos Tovar

        Args:
            entity_startpoint (dict): id de posicion de cada entidad
            id2text (dict): id de posicion de cada texto
            id2bbox (dict): id de posicion de bboxes
        
        Returns:
            entity_startpoint (dict): id de posicion de cada entidad
        """

        def break_group(bbox1, bbox2, limit_y = 10):
            """
            Esta funcion valida si el texto pertenece a una linea. bbox: x1, y1, x2, y2

            Elaborado por: Juan Carlos Tovar
            
            Args:
                bbox1 (list): bbox 1
                bbox2 (list): bbox 2
                limit_y (int): umbral de distancia vertical para considerar que el texto pertenece a una linea
            
            Returns: 
                True si el texto pertenece a una linea, False en caso contrario
            """
            # print(bbox1, bbox2)
            # print(abs(bbox1[1] - bbox2[1]), abs(bbox1[1] - bbox2[1])<limit_y)
            # print(abs(bbox1[3] - bbox2[1]), abs(bbox1[3] - bbox2[1])<limit_y)
            if abs(bbox1[1] - bbox2[1])<=limit_y:
                return False
            if abs(bbox1[3] - bbox2[1])<=limit_y:
                return False
            
            return True

        ent = ''
        for k in self.entidades[1:3]:
            if self.entity_startpoint[k] != -1:
                ent = k
        # print(ent)
        if ent:
            start = self.entity_startpoint[ent]
            end = self.entity_startpoint['DE'] if self.entity_startpoint['DE'] != -1 else self.n_texts
            # print(start, end)
            bbox1 = self.id2bbox[start]
            for i in range(start+1, end):
                bbox2 = self.id2bbox[i]
                
                if break_group(bbox1, bbox2, limit_y = self.y_limit):
                    #print(i, bbox2)
                    self.entity_startpoint['CUERPO'] = i
                    break
                bbox1 = bbox2
        else:
            print("No hay ['ASUNTO', 'REF']")
            first_20_items = list(self.id2text.items())[0:20]
            pattern = re.compile(r'PRESEN\.|PRESENTE', flags=re.IGNORECASE)
            matching_items = [(key, value) for key, value in first_20_items if pattern.search(value)]
            self.entity_startpoint['CUERPO'] = matching_items[0][0]            


#%%Main
if __name__=="__main__":
    dict_doc_ent = {}
    path_error=r'C:\Users\jjuua\VSCode\mef_proyectos\entregables\std_project\log\error_pdf_class_log.txt'
    memo_error_path = r'C:\Users\jjuua\VSCode\mef_proyectos\entregables\data\MEMO_ERROR\ITER2'
    
    logging.basicConfig(filename=path_error, level=logging.ERROR)
    path_sample = r'C:\Users\jjuua\VSCode\mef_proyectos\entregables\productos\P06\pdf_sample'
    path_sample = r'C:\Users\jjuua\VSCode\mef_proyectos\entregables\data\MEMO_TEST\carta'
    path_entity = r'C:\Users\jjuua\VSCode\mef_proyectos\entregables\std_project\data\entitity'
    path_sample = path_sample+r'\0542222020019590000525139963010202005151614352.pdf'
    try:
        d = DOCUMENT(pdf_path=path_sample)
        if d.status:
            if d.digital:
                doc_ocr = DOCUMENT_DIGITAL_PARSER(pdf_path=path_sample, n_pages=d.num_pages)
                if len(doc_ocr.text)==0:
                    doc_ocr = DOCUMENT_SCANNED_PARSER(pdf_path=path_sample, n_pages=d.num_pages)
                    d.digital = False
            else:
                doc_ocr = DOCUMENT_SCANNED_PARSER(pdf_path=path_sample, n_pages=d.num_pages)
                if len(doc_ocr.text)==0:
                    doc_ocr = DOCUMENT_DIGITAL_PARSER(pdf_path=path_sample, n_pages=d.num_pages)
                    d.digital = True
        
            if doc_ocr.status and len(doc_ocr.text)>10:
                image = doc_ocr.plot_with_bounding_boxes(d.digital, doc_ocr.id2bbox, doc_ocr.page_start, doc_ocr.end_page)
                if doc_ocr.doc_type != 'OTROS':
                    doc_ent = DOCUMENT_ENTITIES(id2text=doc_ocr.id2text, id2bbox=doc_ocr.id2bbox)
                    dict_doc_ent = doc_ent.entities_content
                else:
                    dict_doc_ent['FECHA_EXTRACCION'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                if d.digital:
                    dict_doc_ent['TIPO_DOC'] = "Documento Digital"
                else:
                    dict_doc_ent['TIPO_DOC'] = "Documento Escaneado"
                    dict_doc_ent['CONFIANZA_MEDIA_OCR'] = doc_ocr.avg_confidence
                dict_doc_ent['CLASE_DOC'] = doc_ocr.doc_type
                dict_doc_ent['NUM_TOTAL_PAGINAS'] = d.num_pages
                
                json_file = path_entity+'\\'+path_sample.split('\\')[-1].split('.pdf')[0]+'.json'
                d.export_json_entities(dict_doc_ent, json_file)
            else:
                print("Error in DOCUMENT PARSER Sub Class")
                dict_doc_ent['FECHA_EXTRACCION'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if d.digital:
                    dict_doc_ent['TIPO_DOC'] = "Documento Digital"
                else:
                    dict_doc_ent['TIPO_DOC'] = "Documento Escaneado"
                dict_doc_ent['CLASE_DOC'] = "SIN TEXTO"
                dict_doc_ent['NUM_TOTAL_PAGINAS'] = d.num_pages
                json_file = path_entity+'\\'+path_sample.split('\\')[-1].split('.pdf')[0]+'.json'
                d.export_json_entities(dict_doc_ent, json_file)
        else:
            print("Error in DOCUMENT Class")
    except Exception as e:
        with open(path_error, 'a') as error_file:
             traceback_str = traceback.format_exc()
             print(e)
             logging.error(f"Error processing file F: {path_sample}; Error: {str(e)}\n{traceback_str}")
             name_pdf = path_sample.split('\\')[-1].split('.pdf')[0]
             shutil.copy(path_sample, memo_error_path + '\\' + name_pdf + "_error.pdf")
