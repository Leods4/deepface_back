import os
import re
import tempfile
import uuid
import shutil
import numpy as np
import uvicorn
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# --- Configurações do Banco Vetorial (ChromaDB) ---
import chromadb
from chromadb.config import Settings

# Inicializa o ChromaDB para salvar os dados localmente na pasta "./chroma_db"
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Cria ou carrega as coleções de vetores. A métrica "cosine" é ideal para biometria.
colecao_rostos = chroma_client.get_or_create_collection(
    name="rostos", metadata={"hnsw:space": "cosine"}
)
colecao_tatuagens = chroma_client.get_or_create_collection(
    name="tatuagens", metadata={"hnsw:space": "cosine"}
)

# Diretório local para salvar as imagens (substituindo o BYTEA do PostgreSQL)
DIR_IMAGENS = "./banco_imagens"
os.makedirs(os.path.join(DIR_IMAGENS, "rostos"), exist_ok=True)
os.makedirs(os.path.join(DIR_IMAGENS, "tatuagens"), exist_ok=True)

# --- Configurações da API ---
app = FastAPI(title="API de Biometria (Otimizada com ChromaDB)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES_PER_REQUEST = 5
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# --- Funções Auxiliares Matemáticas e de Modelo ---

modelo_tatuagem = None

def carregar_modelo_tatuagem():
    global modelo_tatuagem
    if modelo_tatuagem is None:
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import GlobalAveragePooling2D
        
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base.output)
        modelo_tatuagem = Model(inputs=base.input, outputs=x)
    return modelo_tatuagem

def extrair_vetor_tatuagem(caminho_imagem: str):
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.efficientnet import preprocess_input

    model = carregar_modelo_tatuagem()
    img = keras_image.load_img(caminho_imagem, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    embedding = model.predict(x, verbose=0)[0]
    
    norma = np.linalg.norm(embedding)
    vetor_normalizado = embedding / norma if norma > 0 else embedding
    return vetor_normalizado.tolist()

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

# ==================== ROTAS DE ROSTO (DEEPFACE) ====================

@app.post("/api/cadastrar")
def cadastrar_usuario(files: List[UploadFile] = File(...), nome: str = Form(...)):
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Limite de {MAX_IMAGES_PER_REQUEST} imagens.")
    try:
        nome_sanitizado = sanitizar_nome(nome)
    except ValueError:
        raise HTTPException(status_code=400, detail="Nome inválido.")

    resultados = []
    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            continue
        
        caminho_tmp = salvar_temporario(file)
        
        try:
            # Extrai o vetor
            representacoes = DeepFace.represent(img_path=caminho_tmp, model_name="ArcFace", enforce_detection=False)
            vetor_rosto = representacoes[0]["embedding"]
            
            # Salva a imagem fisicamente no disco
            id_unico = str(uuid.uuid4())
            extensao = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            caminho_final = os.path.join(DIR_IMAGENS, "rostos", f"{nome_sanitizado}_{id_unico}.{extensao}")
            shutil.copy2(caminho_tmp, caminho_final)
            
            # Salva o vetor e os metadados no ChromaDB
            colecao_rostos.add(
                embeddings=[vetor_rosto],
                metadatas=[{"nome_usuario": nome_sanitizado, "caminho_imagem": caminho_final}],
                ids=[id_unico]
            )
            
            resultados.append({"arquivo": file.filename, "status": "sucesso"})
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": str(e)})
        finally:
            os.remove(caminho_tmp)

    return {"resultados": resultados}

@app.post("/api/reconhecer")
def reconhecer_imagens(files: List[UploadFile] = File(...)):
    # No ChromaDB, a distância de Cosseno vai de 0.0 (idêntico) a 1.0 (diferente)
    THRESHOLD_ARCFACE = 0.68 
    resultados = []

    # Verifica se a coleção está vazia
    if colecao_rostos.count() == 0:
        return {"resultados": [{"arquivo": f.filename, "status": "falha", "mensagem": "Banco vazio."} for f in files]}

    for file in files:
        caminho_tmp = salvar_temporario(file)
        try:
            representacoes = DeepFace.represent(img_path=caminho_tmp, model_name="ArcFace", enforce_detection=True)
            vetor_query = representacoes[0]["embedding"]
            
            # Busca NATIVA do ChromaDB (Ultra Rápida)
            resultado_busca = colecao_rostos.query(
                query_embeddings=[vetor_query],
                n_results=1 # Traz apenas o mais próximo
            )
            
            distancia = resultado_busca['distances'][0][0]
            
            if distancia <= THRESHOLD_ARCFACE:
                nome_identificado = resultado_busca['metadatas'][0][0]['nome_usuario']
                resultados.append({
                    "arquivo": file.filename, 
                    "status": "sucesso", 
                    "nome_identificado": nome_identificado, 
                    "distancia": round(distancia, 4)
                })
            else:
                resultados.append({"arquivo": file.filename, "status": "falha", "mensagem": "Sem correspondência."})
                
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": str(e)})
        finally:
            os.remove(caminho_tmp)
            
    return {"resultados": resultados}

# ==================== ROTAS DE TATUAGEM (EFFICIENTNET) ====================

@app.post("/api/cadastrar-tatuagem")
def cadastrar_tatuagem(files: List[UploadFile] = File(...), nome: str = Form(...)):
    try:
        nome_sanitizado = sanitizar_nome(nome)
    except ValueError:
        raise HTTPException(status_code=400, detail="Nome inválido.")

    resultados = []
    for file in files:
        caminho_tmp = salvar_temporario(file)
        
        try:
            vetor_tattoo = extrair_vetor_tatuagem(caminho_tmp)
            
            id_unico = str(uuid.uuid4())
            extensao = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            caminho_final = os.path.join(DIR_IMAGENS, "tatuagens", f"{nome_sanitizado}_{id_unico}.{extensao}")
            shutil.copy2(caminho_tmp, caminho_final)
            
            colecao_tatuagens.add(
                embeddings=[vetor_tattoo],
                metadatas=[{"nome_usuario": nome_sanitizado, "caminho_imagem": caminho_final}],
                ids=[id_unico]
            )
            
            resultados.append({"arquivo": file.filename, "status": "sucesso"})
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": str(e)})
        finally:
            os.remove(caminho_tmp)

    return {"resultados": resultados}

@app.post("/api/reconhecer-tatuagem")
def reconhecer_tatuagem(files: List[UploadFile] = File(...)):
    THRESHOLD_TATTOO = 0.4 
    resultados = []

    if colecao_tatuagens.count() == 0:
        return {"resultados": [{"arquivo": f.filename, "status": "falha", "mensagem": "Banco vazio."} for f in files]}

    for file in files:
        caminho_tmp = salvar_temporario(file)
        try:
            vetor_query = extrair_vetor_tatuagem(caminho_tmp)
            
            resultado_busca = colecao_tatuagens.query(
                query_embeddings=[vetor_query],
                n_results=1
            )
            
            distancia = resultado_busca['distances'][0][0]
            
            if distancia <= THRESHOLD_TATTOO:
                nome_identificado = resultado_busca['metadatas'][0][0]['nome_usuario']
                resultados.append({
                    "arquivo": file.filename, 
                    "status": "sucesso", 
                    "nome_identificado": nome_identificado, 
                    "distancia": round(distancia, 4)
                })
            else:
                resultados.append({"arquivo": file.filename, "status": "falha", "mensagem": "Sem correspondência."})
                
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": str(e)})
        finally:
            os.remove(caminho_tmp)
            
    return {"resultados": resultados}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
