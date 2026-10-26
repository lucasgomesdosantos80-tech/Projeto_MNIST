from pydantic import BaseModel, Field
from typing import List

class MNISTInput(BaseModel):
    # Espera uma lista de 784 valores numéricos (28x28 pixels) correspondentes à imagem do dígito
    pixels: List[float] = Field(
        ..., 
        description="Lista contendo exatamente 784 valores de pixels normalizados entre 0 e 1."
    )

class MNISTResponse(BaseModel):
    predicao: int = Field(..., description="O dígito previsto pelo modelo MLP.")
    probabilidade: float = Field(..., description="A confiança/probabilidade da predição.")