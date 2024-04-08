# 🚀🤖 Ventanilla Cognitiva 🤖🚀

[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![fastapi](https://img.shields.io/badge/FastAPI-009485?style=for-the-badge&logo=fastapi&logoColor=white)](https://img.shields.io/badge/FastAPI-3776AB?style=for-the-badge&logo=fastapi&logoColor=white)
[![ml](https://img.shields.io/badge/Machine-Learning-brightgreen)](https://img.shields.io/badge/Machine-Learning-brightgreen)
[![ocr](https://img.shields.io/badge/OCR-blue)](https://img.shields.io/badge/OCR-blue)
[![nlp](https://img.shields.io/badge/NLP-orange)](https://img.shields.io/badge/NLP-orange)

## Descripción del Proyecto
Proyecto de Analítica Avanzada para la Derivación Automática en el Sistema de Gestión Documental Digital (SGDD)

## Tabla de Contenido
1. [Resumen del Proyecto](#resumen)
    - [1.1. Descripción del conjunto de datos](#conjunto-de-datos)
2. [Stack Tecnológico](#tecnologia)
3. [Entregables](#entregables)
4. [Instalación](#instalacion)
    - [4.1. Instalación local](#instalacion-local)
    - [4.2. Instalación con Docker](#instalacion-docker)
6. [Ejecución](#ejecucion)
7. [API Endpoints](#api-endpoints)
8. [Uso de la Aplicación](#uso)
9.  [Autores](#contacto)

## 1. Resumen del Proyecto <a name="resumen"></a>
- El proyecto de derivación automática gira en torno a dos modelos de aprendizaje automático diseñado para deriviar a una de las direcciones de la AFSP.
- El proyecto proporciona una solución integral de tecnologías open source, como una API con FastAPI que permite el despliegue en una arquitectura de microservicios.
- Para simplificar la implementación y el uso, el proyecto incluye un archivo Docker que agiliza el proceso de configuración y garantiza la instalación de las dependencias necesarias. Esto facilita la implementación del modelo de predicción de sepsis en diversos entornos, tanto locales como en la nube.
- Se proporciona documentación detallada y ejemplos prácticos para guiar a los usuarios en la utilización efectiva del modelo de derivación. La documentación cubre instrucciones de instalación y pautas de uso de la API.

### 1.1. Descripción del conjunto de datos <a name="conjunto-de-datos"></a>
| Nombre de la Columna | Atributo/Objetivo | Descripción |
| -------------------- | ----------------- | ----------- |
| nro_hoja_ruta        | N/A               | ID del expediente |
| fecha_creacion       | N/A               | Fecha de creación del expediente |
| asunto               | Atributo 1        | Asunto del expediente. Dato estructurado proveniente de la base de datos del sistema |
| dato_suficiente      | N/A               | Regla para determinar que modelo utilizar. Valores: {0, 1} |
| cuerpo               | Atributo 2        | Cuerpo del expediente. Dato no estructurado proveniente del documento pdf principal |
| idunidad_organica    | Target            | Identificador de las direcciones de la AFSP, más la clase OTROS |

## 2. Stack Tecnológico <a name="tecnologia"></a>
| Tecnología         | Versión   |
| ------------------ | --------- |
| Python             | 3.11.0    |
| fastapi[all]       | 0.104.1   |
| uvicorn[standard]  | 0.24.0    |
| numpy              | 1.23.5    |
| pandas             | 2.0.3     |
| scikit-learn       | 1.4.0     |
| jinja2             | 3.1.2     |
| connectorx         | 0.3.2     |
| pdfminer.six       | 20221105  |
| pdf2image          | 1.16.3    |
| easyocr            | 1.7.0     |
| PyPDF2             | 3.0.1     |
| thefuzz            | 0.20.0    |
| opencv-python-headless | 4.8.1.78 |
| tqdm               | 4.66.1    |
| nltk               | 3.8.1     |
| spacy              | 3.7.2     |
| unidecode          | 1.3.7     |
| gensim             | 4.3.2     |
| xgboost            | 2.0.3     |
| pyarrow            | 15.0.0    |
| openpyxl           | 3.1.2     |
| pydantic           | 2.6.1     |
| SQLAlchemy         | 2.028     |
| cx-Oracle          | 8.3.0     |

## 3. Entregables <a name="entregables"></a>
1. Un modelo de clasificación entrenado a partir del asunto del expediente (BD)
2. Un modelo de clasificación entrenado a partir del cuerpo del expediente (PDF)
3. Una aplicación de API construida con FastAPI
4. Un archivo Docker para facilitar la implementación

## 4. Instalación <a name="instalacion"></a>
### 4.1. Instalación local <a name="instalacion-local"></a>
Se necesita [`Python 3`](https://www.python.org/downloads/) en el sistema.
1. **Para clonar el repositorio ejecutar lo siguiente:**

        git clone https://csxghhv.mef.gob.pe/MEF-Analitica/Ventanilla_cognitiva.git

---

2. **Navegar a la raíz del proyecto:**

        cd Ventanilla_cognitiva

---

3. **Editar el archivo `variables.json`**, las variables `datalake_landing_path`: ruta de la tabla del datalake donde se almacenan los archivos pre procesados en pormato .json, `schema_name_predict_output`: nombre de esquema de BD (predicciones), `level_deploy`: nivel de despliegue {'dev', 'test', 'prod'} y `ftp_name`: nombre del Servidor FTP.

---

4. **Configurar Poppler:**
    - Windows:

            tar -xf .\comp\Poppler-23.11.0-0.zip -C .\src\lib\

---

5. **Configurar Instant client para Oracle DB:**
    - Windows:
        
        Descomprimir `/comp/instantclient-basic-windows.x64-19.22.0.0.0dbru.zip` en una carpeta (de preferencia en una unidad diferente de la C:), posteriormente agregar la ruta descomprimida en la variable de sistema `Path` y luego aceptar y guardar cambios.

---

6. **Crear un nuevo entorno virtual y activarlo:**
    - Windows:

            python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -r requirements.txt; python -m spacy download es_core_news_sm

    - Linux & MacOs:
    
            python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -r requirements.txt; python -m spacy download es_core_news_sm

---

Las dos líneas de comando tienen la misma estructura. Conectan varios comandos usando el símbolo `;`, pero se puede ejecutar manualmente uno tras otro.

* **Crear el entorno virtual de Python** que aísla las bibliotecas necesarias del proyecto para evitar conflictos;
* **Activar el entorno virtual de Python** para que el kernel y las bibliotecas de Python sean los del entorno aislado;
* **Actualizar pip, el administrador de bibliotecas/paquetes instalados** para tener la versión más actualizada que funcionará correctamente;
* **Instalar las bibliotecas/paquetes requeridos** enumerados en el archivo requirements.txt para que puedan ser importados al script de Python
* **Descargar el paquete de es_core_news_sm con spacy**

---

7. **Configurar credenciales de usuario de base de datos:**
Ejecutar el programa `python src/encripta_credencial.py` e ingresar los parámetros del usuario de base de datos que insertará las predicciones.

![save_credential](https://csxghhv.mef.gob.pe/storage/user/47/files/eaed3abd-3030-4ff2-9d78-5661c1b98158)

---

8. **Configurar credenciales de servidor FTP:**
Ejecutar el programa `python src/encripta_credencial.py` e ingresar los parámetros del usuario del servidor FTP.

![save_credential_ftp](https://csxghhv.mef.gob.pe/storage/user/47/files/93d848ea-15c9-493d-b9b1-81c995b39d15)


### 4.2. Instalación con Docker <a name="instalacion-docker"></a>
1. **Para clonar el repositorio ejecutar lo siguiente:**

        git clone https://csxghhv.mef.gob.pe/MEF-Analitica/Ventanilla_cognitiva.git

2. **Navegar a la raíz del proyecto:**

        cd Ventanilla_cognitiva

3. **Seguir los pasos 3, 7 y 8 de la [Instalación local](#instalacion-local).**

4. **Utilizar el archivo `Dockerfile`** para instalar y dejar en ejecución la solución

---

## 5. Ejecución <a name='ejecucion'></a>
Para ejecutar la API, seguir estos pasos: Después de haber instalado todos los requisitos.

En la raíz del repositorio, en la terminal:
`root :: <Ventanilla_cognitiva> ...` ejecutar el comando:

            uvicorn src.app.app:app --reload 

o

            python src/app/app.py

Abrir un navegador e ingresar a http://127.0.0.1:8000/docs para acceder a la documentación de la API.

**Opcional. **Para ejecutar la aplicación de prueba en streamlit, seguir los siguientes pasos:
 
   1. Instalar la librería de Streamlit:
    
            pip install streamlit==1.32.0 wordcloud==1.9.3
    
   2. Ejecutar la aplicación:
   En la raíz del repositorio, en la terminal:
`root :: <Ventanilla_cognitiva> ...` ejecutar el comando:
         
            streamlit run src/streamlit_app/streamlit_main.py

---

## 6. API Endpoints <a name="api-endpoints"></a>
1. **/**: Este endpoint muestra un mensaje de bienvenida: "Welcome to the Ventanilla Cognitiva...".
2. **/health**: Comprueba el estado de la API.
3. **/model-info**: Devuelve información sobre el modelo.
4. **/preproc**: Recibe entradas y almacena los datos pre procesados (en formato .json) en el datalake
5. **/predict**: Recibe entradas y devuelve predicción.

---

## 7. Uso de la Aplicación <a name="uso"></a>
Para probar los diversos endpoints de la API utilizando la documentación proporcionada, sigue estos pasos:

1. Comienza accediendo a la documentación de la API, que proporciona información detallada sobre los endpoints disponibles y sus funcionalidades.

2. Encuentra la sección que describe los campos de entrada y los parámetros necesarios para cada endpoint. Especificará el formato de datos esperado, como JSON o datos de formulario, y los campos de entrada necesarios.

3. Introduce los datos de entrada requeridos en los campos de entrada correspondientes o en los parámetros según se especifique en la documentación.

4. Envía la solicitud haciendo clic en el botón "Ejecutar" o utilizando el método adecuado en tu herramienta elegida. La API procesará la solicitud y generará la salida en función de los datos proporcionados.

5. Recupera la respuesta de la API, que contendrá la salida generada. Esta salida puede incluir predicciones, puntuaciones de probabilidad u otra información relevante relacionada con la predicción de sepsis.

6. Repite el proceso para probar diferentes endpoints o varía los datos de entrada para explorar las capacidades de la API. Asegúrate de seguir las pautas de la documentación para cada endpoint para obtener resultados precisos.

---

## 8. Autores :writing_hand: <a name="contacto"></a>

<table>
  <tr>
    <th>Nombre</th>
    <th>LinkedIn</th>
    <th>GitHub</th>
    <th>Hugging Face</th>
  </tr>
  <tr>
    <td>Eng Jhosua Torres</td>
    <td><a href="https://pe.linkedin.com/in/jhos1023/">@JhosuaTorres</a></td>
    <td><a href="https://github.com/jhos1023">@JhosuaTorres</a></td>
    <td><a href="https://huggingface.co">---</a></td>
  </tr>
  <tr>
    <td>MSc Juan Carlos Tovar</td>
    <td><a href="https://www.linkedin.com/in/juan-carlos-tovar-galarreta/">@juan-carlos-tovar-galarreta</a></td>
    <td><a href="https://github.com/JuanTovarGalarreta">@JuanTovarGalarreta</a></td>
    <td><a href="https://huggingface.co/JuanCarlosA">@JuanCarlosA</a></td>
  </tr>

</table>

