# 1. Importação dos módulos do FastAPI e do router do MNIST
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers.mnist_router import router as mnist_router


# 2. Configuração do Ciclo de Vida (Lifespan) da Aplicação
@asynccontextmanager
async def lifespan(app: FastAPI):
  # 2.1. Código executado na inicialização do servidor (Arranque da API)
  print("A carregar o modelo MNIST na inicialização do servidor...")

  # O 'yield' separa o momento de arranque do momento de encerramento
  yield

  # 2.2. Código executado quando o servidor é encerrado (Encerramento da API)
  print("A encerrar a aplicação...")


# 3. Inicialização e Metadados da Aplicação FastAPI
app = FastAPI(
    title="API de classificação MNIST",
    description=(
        "API desenvolvida em FastAPI para predição de dígitos manuscritos"
        " utilizando o modelo MLP."
    ),
    version="1.0.0",
    lifespan=lifespan,  # Regista o ciclo de vida configurado acima
)


# 4. Registo e Inclusão das Rotas (Routers)
# Inclui o router de predição do MNIST dentro da aplicação principal
app.include_router(mnist_router)


# 5. Endpoint Raiz (Página inicial da API)
@app.get("/", summary="Endpoint Raiz")
def read_root():
  return {
      "mensagem": "Bem-vindo à API de classificação MNIST - Módulo 2 SCTEC!",
      "documentação": "/docs",
      "redoc": "/redoc",
  }