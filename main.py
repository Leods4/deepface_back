import os
import re
import shutil
import uuid
import uvicorn
import threading
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

app = FastAPI(title="API de Reconhecimento Facial e de Tatuagens")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "fotos"
TATTOO_DIR = "tatuagens"
MAX_IMAGES_PER_REQUEST = 5
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# Lock para evitar concorrência de leitura/escrita no cache do DeepFace
db_lock = threading.Lock()

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(TATTOO_DIR, exist_ok=True)

# Modelo para Tatuagens (Instanciado de forma preguiçosa/lazy load para não pesar o startup)
modelo_tatuagem = None

def carregar_modelo_tatuagem():
    """Carrega um extrator de características leve baseado em MobileNetV2."""
    global modelo_tatuagem
    if modelo_tatuagem is None:
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import GlobalAveragePooling2D
        
        # Carrega o modelo pré-treinado na ImageNet sem a camada de classificação final
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base.output)
        modelo_tatuagem = Model(inputs=base.input, outputs=x)
    return modelo_tatuagem

def extrair_vetor_tatuagem(caminho_imagem: str):
    """Gera um vetor numérico normalizado representando as texturas/formas da imagem."""
    import numpy as np
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = carregar_modelo_tatuagem()
    img = keras_image.load_img(caminho_imagem, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    embedding = model.predict(x, verbose=0)[0]
    norma = np.linalg.norm(embedding)
    return embedding / norma if norma > 0 else embedding

def buscar_tatuagem_proxima(caminho_query: str, threshold: float = 0.5):
    """Compara o vetor da imagem enviada com todas as tatuagens salvas no banco."""
    import numpy as np
    vetor_query = extrair_vetor_tatuagem(caminho_query)
    
    melhor_usuario = None
    melhor_distancia = float('inf')
    melhor_caminho = None
    
    for raiz, _, arquivos in os.walk(TATTOO_DIR):
        for arquivo in arquivos:
            ext = arquivo.split('.')[-1].lower()
            if ext in ['jpg', 'jpeg', 'png']:
                caminho_db = os.path.join(raiz, arquivo)
                try:
                    vetor_db = extrair_vetor_tatuagem(caminho_db)
                    # Distância de Cosseno: 1.0 - Produto Escalar (vetores já normalizados)
                    distancia = 1.0 - float(np.dot(vetor_query, vetor_db))
                    
                    if distancia < melhor_distancia:
                        melhor_distancia = distancia
                        melhor_caminho = caminho_db
                        melhor_usuario = Path(caminho_db).parent.name
                except Exception as e:
                    print(f"Erro ao processar vetor da imagem {caminho_db}: {e}")
                    
    if melhor_usuario and melhor_distancia <= threshold:
        return melhor_usuario, melhor_distancia, melhor_caminho.replace('\\', '/')
    return None, None, None

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

# ==================== ROTAS DE ROSTO (DEEPFACE) ====================

@app.post("/api/cadastrar")
def cadastrar_usuario(files: List[UploadFile] = File(...), nome: str = Form(...)):
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
            resultados.append({"arquivo": file.filename, "status": "sucesso", "mensagem": f"Salvo com sucesso para o usuário '{nome_pasta}'."})
        except Exception:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Erro ao salvar arquivo na API."})

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

        extensao = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        caminho_temporario = f"temp_{uuid.uuid4().hex}.{extensao}"
        
        try:
            with open(caminho_temporario, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            with db_lock:
                resultado = DeepFace.find(img_path=caminho_temporario, db_path=BASE_DIR, model_name="VGG-Face", enforce_detection=True, silent=True)
            
            if len(resultado) > 0 and not resultado[0].empty:
                df_resultado = resultado[0]
                arquivo_encontrado = df_resultado['identity'][0]
                valor_confianca = float(df_resultado['distance'][0] if 'distance' in df_resultado.columns else df_resultado.iloc[0, -1])
                caminho_completo = str(arquivo_encontrado).replace('\\', '/')
                nome_identificado = Path(caminho_completo).parent.name

                resultados.append({
                    "arquivo": file.filename, "status": "sucesso", "nome_identificado": nome_identificado,
                    "caminho_imagem": caminho_completo, "distancia": round(valor_confianca, 4)
                })
            else:
                resultados.append({"arquivo": file.filename, "status": "falha", "mensagem": "Nenhuma correspondência facial encontrada."})
        except ValueError: 
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Nenhum rosto detectado na imagem."})
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Erro interno durante o reconhecimento."})
            print(f"Erro no DeepFace: {e}") 
        finally:
            if os.path.exists(caminho_temporario):
                os.remove(caminho_temporario)
                
    return {"resultados": resultados}

# ==================== ROTAS DE TATUAGEM (MOBILENETV2) ====================

@app.post("/api/cadastrar-tatuagem")
def cadastrar_tatuagem(files: List[UploadFile] = File(...), nome: str = Form(...)):
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"O limite máximo é de {MAX_IMAGES_PER_REQUEST} imagens.")
    if not nome or not nome.strip():
        raise HTTPException(status_code=400, detail="Nome não fornecido.")

    try:
        nome_pasta = sanitizar_nome_pasta(nome)
    except ValueError:
        raise HTTPException(status_code=400, detail="O nome fornecido contém caracteres inválidos.")

    caminho_pasta = os.path.join(TATTOO_DIR, nome_pasta)
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
            resultados.append({"arquivo": file.filename, "status": "sucesso", "mensagem": f"Tatuagem salva com sucesso para o usuário '{nome_pasta}'."})
        except Exception:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Erro ao salvar tatuagem na API."})

    return {"resultados": resultados}

@app.post("/api/reconhecer-tatuagem")
def reconhecer_tatuagem(files: List[UploadFile] = File(...)):
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"O limite máximo é de {MAX_IMAGES_PER_REQUEST} imagens.")

    resultados = []
    for file in files:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Formato não suportado."})
            continue

        extensao = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        caminho_temporario = f"temp_tattoo_{uuid.uuid4().hex}.{extensao}"
        
        try:
            with open(caminho_temporario, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            nome_identificado, distancia, caminho_completo = buscar_tatuagem_proxima(caminho_temporario)
            
            if nome_identificado:
                resultados.append({
                    "arquivo": file.filename, "status": "sucesso", "nome_identificado": nome_identificado,
                    "caminho_imagem": caminho_completo, "distancia": round(distancia, 4)
                })
            else:
                resultados.append({"arquivo": file.filename, "status": "falha", "mensagem": "Nenhuma correspondência de tatuagem encontrada."})
        except Exception as e:
            resultados.append({"arquivo": file.filename, "status": "erro", "mensagem": "Erro ao processar análise da tatuagem."})
            print(f"Erro no Reconhecimento de Tatuagem: {e}")
        finally:
            if os.path.exists(caminho_temporario):
                os.remove(caminho_temporario)
                
    return {"resultados": resultados}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
