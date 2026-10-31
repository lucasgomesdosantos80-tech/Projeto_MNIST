# 1. Importação das classes base e utilitários de validação do Pydantic
from pydantic import BaseModel, Field


# 2. Definição do Schema/Modelo estruturado para a Resposta da API
class MNISTResponse(BaseModel):
  # 2.1. Campo correspondente ao dígito previsto pelo modelo (inteiro de 0 a 9)
  predicao: int = Field(
      ..., description="O dígito previsto pelo modelo MLP (0 a 9)."
  )

  # 2.2. Campo correspondente ao índice de confiança/probabilidade da predição (decimal)
  probabilidade: float = Field(
      ..., description="A confiança/probabilidade da predição."
  )