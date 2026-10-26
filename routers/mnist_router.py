from fastapi import APIRouter, HTTPException
from schemas.mnist_schema import MNISTInput, MNISTResponse
from services.mnist_service import mnist_service

router = APIRouter(prefix="/mnist", tags=["Classificação MNIST"])

@router.get("/health", summary="Checagem de integridade operacional")
def health_check():
    return{
        "status": "online",
        "message": "Serviço ativo e modelo de Machine Learning carregado na memória RAM"
    }

@router.post("/predict", response_model=MNISTResponse, summary="Prediz o dígito manuscrito com base nos pixels")
def predict_digit(data: MNISTInput):
    try:
        # Garante que a lista possui exatamente 784 elementos (28x28 pixels)
        if len(data.pixels) != 784:
            raise HTTPException(
                status_code=400, 
                detail="A lista de pixels deve conter exatamente 784 valores."
            )
        
        # Executa a predição através do serviço
        result = mnist_service.predict(data.pixels)
        
        return MNISTResponse(
            predicao=result["predicao"],
            probabilidade=result["probabilidade"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))