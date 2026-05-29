# Documentação da API de Biometria (Facial e Tatuagens)

Esta é uma API RESTful de alto desempenho desenvolvida em **Python** utilizando **FastAPI**. Ela permite o cadastro e o reconhecimento biométrico de usuários através de análise facial e de tatuagens, utilizando redes neurais avançadas e um banco de dados vetorial local (ChromaDB) para buscas em milissegundos.

## 🛠 Tecnologias Utilizadas

* **FastAPI:** Framework web principal para roteamento e endpoints.
* **ChromaDB:** Banco de dados vetorial (*Vector Database*) para armazenamento local e busca por similaridade (Distância de Cosseno).
* **DeepFace (ArcFace):** Modelo de estado da arte para extração de características faciais (vetores de 512 dimensões).
* **TensorFlow / EfficientNetB0:** Modelo de visão computacional para extração de características de tatuagens (vetores de 1280 dimensões).
* **Uvicorn:** Servidor ASGI para rodar a aplicação.

---

## ⚙️ Instalação e Configuração

### 1. Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado. É recomendável o uso de um ambiente virtual (`venv`).

### 2. Instalação das Dependências
Execute o comando abaixo no terminal para instalar todas as bibliotecas necessárias:

```bash
pip install fastapi uvicorn python-multipart deepface chromadb tensorflow numpy
```
*(Nota: O `python-multipart` é essencial para que o FastAPI consiga receber formulários e arquivos de imagem via requisições POST).*

### 3. Executando o Servidor
Para iniciar a API em modo de desenvolvimento (com *reload* automático), execute:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

A documentação interativa (Swagger UI) estará disponível automaticamente em: **http://127.0.0.1:8000/docs**

---

## 📂 Arquitetura de Armazenamento Local

A API não requer serviços externos. Ao rodar a aplicação e realizar os primeiros cadastros, ela criará automaticamente duas pastas na raiz do projeto:

* `./chroma_db/`: Contém os arquivos do banco de dados vetorial SQLite/Parquet (Embeddings e Metadados).
* `./banco_imagens/`:
    * `/rostos/`: Onde as imagens faciais físicas são salvas de forma persistente.
    * `/tatuagens/`: Onde as imagens de tatuagens físicas são salvas.

---

## 📡 Endpoints da API

Abaixo estão os detalhes de cada rota. Todas as requisições que enviam imagens devem usar o `Content-Type: multipart/form-data`. O limite é de **5 imagens por requisição**. Formatos aceitos: JPG, JPEG e PNG.

### 1. Cadastro de Rosto
Extrai as características faciais da imagem e as salva no banco vetorial atreladas a um nome.

* **URL:** `/api/cadastrar`
* **Método:** `POST`
* **Parâmetros (Form-Data):**
    * `files` (Array de Arquivos): As imagens contendo os rostos.
    * `nome` (String): Nome do usuário a ser cadastrado.
* **Respostas:**
    * `200 OK`: Retorna um JSON detalhando o status de cada arquivo enviado.
    * `400 Bad Request`: Se o limite de imagens for excedido ou o nome contiver caracteres inválidos.

### 2. Reconhecimento de Rosto
Compara a imagem enviada com o banco de dados e retorna o usuário mais próximo (se a similaridade for aceitável).

* **URL:** `/api/reconhecer`
* **Método:** `POST`
* **Parâmetros (Form-Data):**
    * `files` (Array de Arquivos): As imagens a serem reconhecidas.
* **Limiar (Threshold):** `0.68` (Distância de Cosseno).
* **Retorno de Sucesso:**
    ```json
    {
      "resultados": [
        {
          "arquivo": "foto_camera.jpg",
          "status": "sucesso",
          "nome_identificado": "joao_silva",
          "distancia": 0.3412
        }
      ]
    }
    ```

### 3. Cadastro de Tatuagem
Extrai características visuais da tatuagem usando *EfficientNetB0* e salva no banco vetorial.

* **URL:** `/api/cadastrar-tatuagem`
* **Método:** `POST`
* **Parâmetros (Form-Data):**
    * `files` (Array de Arquivos): As imagens das tatuagens.
    * `nome` (String): Nome do usuário ou identificador da tatuagem.
* **Respostas:**
    * `200 OK`: Retorna o status de sucesso para cada imagem cadastrada.

### 4. Reconhecimento de Tatuagem
Compara a tatuagem enviada com as tatuagens registradas no banco.

* **URL:** `/api/reconhecer-tatuagem`
* **Método:** `POST`
* **Parâmetros (Form-Data):**
    * `files` (Array de Arquivos): As imagens a serem pesquisadas.
* **Limiar (Threshold):** `0.4` (Distância de Cosseno - pode requerer ajuste fino conforme o dataset em produção).
* **Retorno de Sucesso:** JSON com o nome identificado e a distância calculada.

---

## 🧠 Lógica de Limiares (Thresholds) e Distância

O sistema utiliza a **Distância de Cosseno (Cosine Distance)** calculada nativamente pelo ChromaDB. 
* O valor de distância varia de `0.0` a `1.0`. 
* Quanto **menor** o valor, mais idênticas são as imagens (Ex: `0.0` significa uma correspondência matematicamente perfeita).
* O valor de corte determina a tolerância a falsos positivos. Valores acima do limiar configurado nas rotas (`0.68` para rostos e `0.4` para tatuagens) retornarão "Sem correspondência", evitando que pessoas erradas sejam aprovadas.
