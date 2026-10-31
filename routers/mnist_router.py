# 1. Importação das bibliotecas e dependências do FastAPI e da camada de serviço
from fastapi import APIRouter, File, HTTPException, UploadFile
from services.mnist_service import mnist_service

# 2. Configuração do APIRouter com o prefixo "/mnist" e tags para o Swagger
router = APIRouter(prefix="/mnist", tags=["Classificação MNIST"])


# 3. Rota POST para receber o upload da imagem do dígito manuscrito
@router.post("/predict")
async def prever_digito(file: UploadFile = File(...)):
  try:
    # 3.1. Leitura dos bytes binários da imagem enviada pelo utilizador
    image_bytes = await file.read()

    # 3.2. Chamada do serviço para pré-processamento e inferência do modelo
    resultado = mnist_service.predict_image(image_bytes)

    # 3.3. Retorno bem-sucedido do resultado estruturado
    return resultado

  # 4. Tratamento de Erros e Códigos de Status HTTP
  except RuntimeError as re:
    # 4.1. Erro 500: Falha crítica interna ou problema com o modelo na RAM
    raise HTTPException(
        status_code=500, detail=f"Erro interno no servidor: {str(re)}"
    )

  except Exception as e:
    # 4.2. Erro 400: Falha no formato ou processamento da imagem enviada
    raise HTTPException(
        status_code=400, detail=f"Erro ao processar a imagem: {str(e)}"
    )