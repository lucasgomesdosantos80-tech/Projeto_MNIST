from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers.mnist_router import router as mnist_router

# Define  o ciclo de vida (Lifespan) da aplicação
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executado na inicialização da aplicação (Carrega o modelo na RAM uma única vez)
    print("A carregar o modelo MNIST na inicialização do servidor...")
    # Se necessário, pode chamar aqui a função de carregamento do service
    yield
    # Executado quando a aplicação é encerrada
    print("A encerrar a aplicação...")

app = FastAPI(
    title="API de classificação MNIST",
    description="API desenvolvida em FastAPI para predição de dígitos manuscritos utilizando o modelo MLP.",
    version="1.0.0",
    lifespan=lifespan  # <--- Registra o lifespan aqui
) 

# Inclui o router da predição do MNIST
app.include_router(mnist_router)

@app.get("/", summary="Endpoint Raiz")
def read_root():
    return {
        "mensagem": "Bem-vindo à API de classificação MNIST - Módulo 2 SCTEC!",
        "documentação": "/docs",
        "redoc": "/redoc"
        
    }