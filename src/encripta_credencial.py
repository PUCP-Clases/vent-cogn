#%% TODO Importa librerías
import time
from cryptography.fernet import Fernet
import json
from getpass import getpass
import os
#%% TODO Lectura de variables y credenciales

path_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

var_path = os.path.join(path_project, 'variables.json')

with open(var_path) as json_file:
    var = json.load(json_file)

cred_path = os.path.join(os.path.join(path_project, var['credential_path']),'credentials.json') 

with open(cred_path) as json_file:
    creds = json.load(json_file)
#%%TODO Lectura de llave
key_file_path = os.path.join(os.path.join(path_project, var['key_path']),'filekey.key')

with open(key_file_path, 'rb') as filekey:
    key = filekey.read()

#%% TODO función para crear la cadena de conexion
def get_engine(e):
    options = {1: 'oracle', 2: 'postgresql', 3: 'mysql', 4:'mssql', 5:'ftp'}
    if e in options:
        return options[e]
    else:
        print("Invalid choice.")

def get_environment(e):
    options = {1: 'dev', 2: 'test', 3: 'prod'}
    if e in options:
        return options[e]
    else:
        print("Invalid choice.")

def get_cred_string(engine, user, pwd, server, port, service):

    if engine == 'ftp':
        return f"{engine}://{user}:{pwd}@{server}:{port}"
    elif engine == 'mssql':
        return f"{engine}://{user}:{pwd}@{server}:{port}/{service}?encrypt=true&trusted_connection=true"
    else:
        return f"{engine}://{user}:{pwd}@{server}:{port}/{service}"

def encrypt_string_conn(cred_string, cred_name, creds, cred_path, key):

    if cred_name in creds.keys():
        print('La cadena ', cred_name, ' ya existe.')
        time.sleep(7)
    else:
        # TODO Encripta cadena
        fernet = Fernet(key)
        enc_cred_string = fernet.encrypt(cred_string.encode('utf-8'))

        print('Cadena', cred_name, ' original: ', cred_string)
        print('Cadena encriptada con llave privada: ', enc_cred_string)

        creds[cred_name] = enc_cred_string.decode('utf-8')

        # Serialización en json
        json_object = json.dumps(creds, indent=4)

        # Escribir en json
        with open(cred_path, "w") as outfile:
            outfile.write(json_object)
        print('Cadena insertada en: ', cred_path)
        time.sleep(7)


#%% TODO Asignación de valores
def main():
    print("***Program for encryptig credentials***")
    print("Please select an option:")
    print("1. Oracle")
    print("2. Postgres")
    print("3. Mysql")
    print("4. MsSQL")
    print("5. FTP")
    choice = int(input("Enter your choice: "))
    engine = get_engine(choice)
    user = str(input("Enter the user: "))
    pwd = getpass("Enter the password: ")
    server = str(input("Enter the server: "))
    port = str(input("Enter the port: "))
    if engine != 'ftp':
        service = str(input("Enter the bd/service: "))
    else:
        service = ''

    cred_string = get_cred_string(engine, user, pwd, server, port, service)
    # TODO esquema y ambiente
    print("***\nValues for credential name***")
    text_name =''
    if engine != 'ftp':
        text_name = "Enter the BD schema (predictions): "
    else:
        text_name = "Enter the FTP Server Name: "
    schema = str(input(text_name))
    print("Please enter the environment:")
    print("1. Development")
    print("2. Test")
    print("3. Production")
    level = int(input("Enter your choice: "))
    level = get_environment(level)
#%% TODO creacion de nueva cadena de conexion
    cred_name = 'engine_' + schema + '_' + level
    encrypt_string_conn(cred_string, cred_name, creds, cred_path, key)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import sys
        print(sys.exc_info()[0])
        import traceback
        print(traceback.format_exc())
        print("Press Enter to continue ...")
        input()