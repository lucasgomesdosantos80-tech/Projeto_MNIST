import joblib
import numpy as np
import os

# Caminho para o modelo guardado na pasta models/
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo_mnist.pkl')

class MNISTService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                print("Modelo MLP carregado com sucesso!")
            else:
                raise FileNotFoundError(f"Ficheiro do modelo não encontrado em: {MODEL_PATH}")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")

    def predict(self, pixels: list) -> dict:
        if self.model is None:
            raise RuntimeError("O modelo não foi carregado corretamente.")
        
        # Converte a lista de pixels para o formato de array 2D que o scikit-learn espera (1 amostra, 784 features)
        data_array = np.array(pixels).reshape(1, -1)
        
        # Realiza a predição do dígito
        prediction = self.model.predict(data_array)
        
        # Tenta obter a probabilidade/confiança, se o modelo suportar predict_proba
        try:
            probabilities = self.model.predict_proba(data_array)
            confidence = float(np.max(probabilities))
        except AttributeError:
            confidence = 1.0  # Caso o modelo não suporte probabilidades nativas

        return {
            "predicao": int(prediction[0]),
            "probabilidade": confidence
        }

# Instância global do serviço para ser injetada nos routers
mnist_service = MNISTService()