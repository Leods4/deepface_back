# app/routes.py
import os
import uuid
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

# Importações dos módulos locais
from app.models import colecao_rostos, colecao_tatuagens, DIR_IMAGENS
from app.services.utils import sanitizar_nome, salvar_temporario
from app.services.vision import extrair_vetor_tatuagem

# --- Configurações da API ---
app = FastAPI(title="API de Biometria (Otimizada e Modularizada)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES_PER_REQUEST = 5
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

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
            representacoes = DeepFace.represent(img_path=caminho_tmp, model_name="ArcFace", enforce_detection=False)
            vetor_rosto = representacoes[0]["embedding"]
            
            id_unico = str(uuid.uuid4())
            extensao = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            caminho_final = os.path.join(DIR_IMAGENS, "rostos", f"{nome_sanitizado}_{id_unico}.{extensao}")
            shutil.copy2(caminho_tmp, caminho_final)
            
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
    THRESHOLD_ARCFACE = 0.68 
    resultados = []

    if colecao_rostos.count() == 0:
        return {"resultados": [{"arquivo": f.filename, "status": "falha", "mensagem": "Banco vazio."} for f in files]}

    for file in files:
        caminho_tmp = salvar_temporario(file)
        try:
            representacoes = DeepFace.represent(img_path=caminho_tmp, model_name="ArcFace", enforce_detection=True)
            vetor_query = representacoes[0]["embedding"]
            
            resultado_busca = colecao_rostos.query(
                query_embeddings=[vetor_query],
                n_results=1
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