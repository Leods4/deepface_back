import os
import re
import shutil
import uuid
import uvicorn
import threading
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# ==========================================
# CONFIGURAÇÕES E INICIALIZAÇÃO
# ==========================================
app = FastAPI(title="API de Reconhecimento Facial")

# Configuração de CORS (Em produção, evite allow_origins=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "fotos"
MAX_IMAGES_PER_REQUEST = 5
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# Lock para evitar que o DeepFace tente ler/escrever o cache simultaneamente
db_lock = threading.Lock()

# Garante que a pasta base existe
os.makedirs(BASE_DIR, exist_ok=True)

# ==========================================
# FUNÇÕES UTILITÁRIAS
# ==========================================
def sanitizar_nome_pasta(nome: str) -> str:
    """Remove caracteres especiais e espaços para evitar Path Traversal."""
    nome_limpo = re.sub(r'[^a-zA-Z0-9_-]', '', nome.replace(" ", "_"))
    if not nome_limpo:
        raise ValueError("Nome inválido.")
    return nome_limpo

def limpar_cache_deepface():
    """Remove os arquivos .pkl para forçar o DeepFace a reindexar as imagens."""
    for item in os.listdir(BASE_DIR):
        if item.endswith(".pkl"):
            try:
                os.remove(os.path.join(BASE_DIR, item))
            except Exception as e:
                print(f"Erro ao remover arquivo de cache {item}: {e}")

# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/api/cadastrar")
def cadastrar_usuario(
    files: List[UploadFile] = File(...),
    nome: str = Form(...)
):
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"O limite máximo é de {MAX_IMAGES_PER_REQUEST} imagens.")
    
    if not nome or not nome.strip():
        raise HTTPException(status_code=400, detail="Nome não fornecido.")

    try:
        nome_pasta = sanitizar_nome_pasta(nome)
    except ValueError:
        raise HTTPException(status_code=400, detail="O nome fornecido contém caracteres inválidos.")

    caminho_pasta = os.path.join(BASE_DIR, nome_pasta)
    os.makedirs(caminho_pasta, exist_ok=True)

    resultados = []

    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Formato não suportado."})
            continue
        
        extensao = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        nome_arquivo_unico = f"{uuid.uuid4().hex}.{extensao}"
        caminho_destino = os.path.join(caminho_pasta, nome_arquivo_unico)
        
        try:
            with open(caminho_destino, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            resultados.append({
                "arquivo": file.filename,
                "status": "sucesso",
                "mensagem": f"Salvo com sucesso na pasta '{nome_pasta}'."
            })
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Erro ao salvar arquivo na API."})

    # Bloqueia a thread apenas na hora de limpar o cache crítico
    with db_lock:
        limpar_cache_deepface()

    return {"resultados": resultados}


@app.post("/api/reconhecer")
def reconhecer_imagens(files: List[UploadFile] = File(...)):
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"O limite máximo é de {MAX_IMAGES_PER_REQUEST} imagens.")

    resultados = []

    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Formato não suportado."})
            continue

        extensao = file.filename.split(".")[-1]
        caminho_temporario = f"temp_{uuid.uuid4().hex}.{extensao}"
        
        try:
            with open(caminho_temporario, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Bloqueia o uso do DeepFace para evitar leitura enquanto um cadastro recria o cache
            with db_lock:
                resultado = DeepFace.find(img_path=caminho_temporario, db_path=BASE_DIR, enforce_detection=True, silent=True)
            
            if len(resultado) > 0 and not resultado[0].empty:
                df_resultado = resultado[0]
                arquivo_encontrado = df_resultado['identity'][0]
                valor_confianca = float(df_resultado['distance'][0] if 'distance' in df_resultado.columns else df_resultado.iloc[0, -1])
                
                caminho_completo = str(arquivo_encontrado).replace('\\', '/')
                resultados.append({
                    "arquivo": file.filename,
                    "status": "sucesso", 
                    "caminho_imagem": caminho_completo,
                    "distancia": round(valor_confianca, 4)
                })
            else:
                resultados.append({
                    "arquivo": file.filename,
                    "status": "falha", 
                    "caminho_imagem": "Desconhecido"
                })
                
        except ValueError: # DeepFace levanta ValueError se não achar rosto
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Nenhum rosto detectado na imagem."})
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Erro interno durante o reconhecimento."})
            print(f"Erro no DeepFace: {e}") # Log no servidor em vez de devolver pro cliente
            
        finally:
            if os.path.exists(caminho_temporario):
                os.remove(caminho_temporario)
                
    return {"resultados": resultados}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)