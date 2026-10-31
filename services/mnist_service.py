# 1. Importação das bibliotecas essenciais para manipulação de imagem, arrays e carregamento do modelo
import io
import cv2
import joblib
import numpy as np
import os
from PIL import Image

# 2. Definição do caminho absoluto para localizar o ficheiro treinado do modelo na pasta "models/"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "modelo_mnist.pkl"
)


# 3. Definição da Classe responsável pela lógica de negócio e inferência do modelo
class MNISTService:

  def __init__(self):
    self.model = None
    # 3.1. Carrega o modelo automaticamente para a memória RAM assim que a classe é instanciada
    self.load_model()

  def load_model(self):
    try:
      # 3.2. Verifica se o ficheiro do modelo existe no disco antes de tentar carregar
      if os.path.exists(MODEL_PATH):
        self.model = joblib.load(MODEL_PATH)
        print("Modelo MLP carregado com sucesso na RAM!")
      else:
        raise FileNotFoundError(
            f"Ficheiro do modelo não encontrado em: {MODEL_PATH}"
        )
    except Exception as e:
      print(f"Erro ao carregar o modelo: {e}")

  def predict_image(self, image_bytes: bytes) -> dict:
    # 3.3. Validação de segurança para garantir que o modelo está residente na RAM
    if self.model is None:
      raise RuntimeError("O modelo não foi carregado corretamente.")

    # 4. Etapa de Pré-processamento da Imagem
    # 4.1. Abre a imagem binária recebida via upload e converte para escala de cinza ("L")
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_array = np.array(img_pil)

    # 4.2. Inversão inteligente de cores (caso o fundo seja claro e o dígito escuro)
    if img_array.mean() > 127:
      img_array = 255 - img_array

    # 4.3. Redimensionamento estrito para o padrão MNIST de 28x28 pixels
    img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

    # 4.4. Normalização dos pixels (escala de 0 a 1) e remodelação para vetor de 784 colunas
    img_normalized = img_resized.astype("float32") / 255.0
    data_array = img_normalized.reshape(1, -1)

    # 5. Etapa de Inferência / Predição
    # 5.1. Realiza a predição do dígito manuscrito usando o modelo MLP
    prediction = self.model.predict(data_array)

    # 5.2. Cálculo e obtenção do grau de confiança/probabilidade da predição
    try:
      probabilities = self.model.predict_proba(data_array)
      confidence = float(np.max(probabilities))
    except AttributeError:
      confidence = 1.0

    # 6. Retorno estruturado do resultado final em formato dicionário
    return {"digito_previsto": int(prediction[0]), "probabilidade": confidence}


# 7. Criação da Instância Global do Serviço
# (Garante que o modelo é carregado na RAM apenas uma vez no arranque da aplicação - Padrão Singleton)
mnist_service = MNISTService()