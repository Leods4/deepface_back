# app/models.py
import os
import chromadb

# --- Configurações de Diretórios Locais ---
DIR_IMAGENS = "./banco_imagens"
os.makedirs(os.path.join(DIR_IMAGENS, "rostos"), exist_ok=True)
os.makedirs(os.path.join(DIR_IMAGENS, "tatuagens"), exist_ok=True)

# --- Configurações do Banco Vetorial (ChromaDB) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Cria ou carrega as coleções de vetores.
colecao_rostos = chroma_client.get_or_create_collection(
    name="rostos", metadata={"hnsw:space": "cosine"}
)

colecao_tatuagens = chroma_client.get_or_create_collection(
    name="tatuagens", metadata={"hnsw:space": "cosine"}
)