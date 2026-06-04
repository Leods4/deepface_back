# 🧬 API de Biometria (Otimizada com ChromaDB)

Uma API RESTful construída com **FastAPI** para reconhecimento biométrico avançado. O sistema realiza a extração e comparação de características físicas (rostos e tatuagens) utilizando modelos de Inteligência Artificial e armazena os embeddings (vetores) em um banco de dados vetorial de alta performance (**ChromaDB**).

---

## 🚀 Recursos

* **Reconhecimento Facial:** Utiliza a biblioteca `DeepFace` (modelo ArcFace) para extrair e comparar características faciais com alta precisão.
* **Reconhecimento de Tatuagens:** Implementa um modelo `EfficientNetB0` (via TensorFlow/Keras) adaptado para extrair vetores de imagens de tatuagens.
* **Busca Vetorial Ultra Rápida:** Integração com `ChromaDB` usando a métrica de distância de Cosseno, garantindo buscas instantâneas mesmo com grandes volumes de dados.
* **Armazenamento Local:** Salva fisicamente as imagens cadastradas de forma organizada, referenciando-as nos metadados do banco vetorial.
* **Arquitetura Modular:** Estrutura baseada em um padrão MVC informal para facilitar a manutenção e escalabilidade.

---

## 📂 Estrutura do Projeto

O projeto adota uma arquitetura modularizada, separando as responsabilidades de inicialização, regras de negócio, IA e banco de dados:

```text
meu_projeto/
│
├── run.py                    # Ponto de entrada (inicia o servidor Uvicorn)
├── chroma_db/                # (Gerado automaticamente) Banco de dados vetorial local
├── banco_imagens/            # (Gerado automaticamente) Imagens salvas (rostos e tatuagens)
│
└── app/
    ├── __init__.py           
    ├── models.py             # Configuração do ChromaDB e diretórios de dados
    ├── routes.py             # Instância do FastAPI e definição dos Endpoints (Controllers)
    │
    └── services/
        ├── __init__.py
        ├── utils.py          # Funções auxiliares (sanitização, arquivos temporários)
        └── vision.py         # Lógica pesada de IA (Modelos TensorFlow/EfficientNet)
```

---

## 🛠️ Tecnologias Utilizadas

* **[FastAPI](https://fastapi.tiangolo.com/):** Framework web rápido para construção da API.
* **[Uvicorn](https://www.uvicorn.org/):** Servidor ASGI para rodar a aplicação.
* **[ChromaDB](https://www.trychroma.com/):** Banco de dados de vetores (Vector Database) operando de forma persistente local.
* **[DeepFace](https://github.com/serengil/deepface):** Framework focado em reconhecimento facial.
* **[TensorFlow / Keras](https://www.tensorflow.org/):** Para o modelo de extração de características de tatuagens (EfficientNet).
* **[NumPy](https://numpy.org/):** Operações matemáticas e normalização de vetores.

---

## ⚙️ Instalação e Execução

### 1. Pré-requisitos
Recomenda-se o uso de **Python 3.9 a 3.11**. Certifique-se de ter o `pip` atualizado.

### 2. Clonar e preparar o ambiente
Crie um ambiente virtual para isolar as dependências pesadas de IA:

```bash
# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
venv\Scripts\activate
# Ativar o ambiente (Linux/Mac)
source venv/bin/activate
```

### 3. Instalar Dependências
Você precisará instalar os pacotes principais do projeto:

```bash
pip install fastapi uvicorn python-multipart chromadb deepface tensorflow numpy
```

*(Nota: Na primeira execução, o DeepFace e o Keras baixarão os pesos dos modelos `ArcFace` e `EfficientNet` automaticamente).*

### 4. Rodar o Servidor
Com as dependências instaladas, inicie a API através do ponto de entrada principal:

```bash
python run.py
```

A API estará disponível em: `http://127.0.0.1:8000`
A documentação interativa (Swagger UI) estará em: `http://127.0.0.1:8000/docs`

---

## 📖 Endpoints da API

Abaixo está o resumo das rotas disponíveis. Acesse o `/docs` da aplicação para testar via interface.

### 👤 Rotas de Rosto

* **`POST /api/cadastrar`**
    * **Descrição:** Cadastra até 5 imagens faciais de um mesmo usuário.
    * **Parâmetros (Form-Data):**
        * `files`: Lista de imagens (jpeg, jpg, png).
        * `nome`: Nome do usuário (será sanitizado para ser usado como pasta/identificador).
    
* **`POST /api/reconhecer`**
    * **Descrição:** Recebe imagens e busca no ChromaDB qual o rosto cadastrado mais semelhante.
    * **Parâmetros (Form-Data):**
        * `files`: Lista de imagens para reconhecimento.
    * **Regra:** Utiliza *Cosine Distance* com threshold de `0.68` para o modelo ArcFace.

### ⚓ Rotas de Tatuagem

* **`POST /api/cadastrar-tatuagem`**
    * **Descrição:** Cadastra imagens de tatuagens, extraindo vetores via EfficientNetB0.
    * **Parâmetros (Form-Data):**
        * `files`: Lista de imagens (jpeg, jpg, png).
        * `nome`: Nome ou identificação da pessoa/tatuagem.

* **`POST /api/reconhecer-tatuagem`**
    * **Descrição:** Busca tatuagens similares no banco de dados.
    * **Parâmetros (Form-Data):**
        * `files`: Lista de imagens para reconhecimento.
    * **Regra:** Utiliza *Cosine Distance* com threshold ajustado para `0.4`.

---

## ⚠️ Observações de Desempenho e Segurança

* **Limpeza de Temporários:** A API utiliza arquivos temporários para processamento, mas garante a deleção automática ao fim de cada request no bloco `finally`.
* **Sanitização:** Entradas de nome de usuário passam por uma função de *Regex* rigorosa para evitar injeção de caracteres que possam corromper os caminhos do sistema operacional.
* **Escalabilidade do ChromaDB:** O banco de dados está rodando em modo local persistente (`PersistentClient`). Para deploy em produção com múltiplos containers, considere migrar para o ChromaDB em modo Servidor (Client-Server).
