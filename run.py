# run.py
import uvicorn

if __name__ == "__main__":
    # Aponta para a instância 'app' dentro de 'app/routes.py'
    uvicorn.run("app.routes:app", host="127.0.0.1", port=8000, reload=True)