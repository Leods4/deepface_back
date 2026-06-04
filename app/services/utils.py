# app/services/utils.py
import re
import tempfile
from fastapi import UploadFile

def sanitizar_nome(nome: str) -> str:
    nome_limpo = re.sub(r'[^a-zA-Z0-9_-]', '', nome.replace(" ", "_"))
    if not nome_limpo:
        raise ValueError("Nome inválido.")
    return nome_limpo

def salvar_temporario(file: UploadFile) -> str:
    extensao = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extensao}") as tmp:
        tmp.write(file.file.read())
        file.file.seek(0) 
        return tmp.name