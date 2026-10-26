# Classificador de Dígitos Manuscritos - MNIST

## 1. Nome e Resumo do Projeto
Este projeto consiste no desenvolvimento e disponibilização de uma API para servir um modelo de Machine Learning capaz de classificar e reconhecer dígitos manuscritos (de 0 a 9) utilizando o conjunto de dados MNIST. O objetivo é automatizar o reconhecimento de imagens com alta precisão através do pré-processamento de dados e disponibilização de inferências em tempo real.

## 2. Origem do Modelo
O modelo preditivo foi desenvolvido e treinado com base nos projetos do curso do SCTEC (Módulo 2), utilizando algoritmos de classificação otimizados para reconhecimento de imagens e exportados para reutilização na API.

## 3. Tecnologias e Técnicas Utilizadas
* **Linguagem:** Python
* **Web Framework:** FastAPI
* **Validação de Dados:** Pydantic
* **Servidor ASGI:** Uvicorn
* **Análise e Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn / Joblib (Pré-processamento, divisão de treino/teste, métricas de avaliação e algoritmos de classificação)
* **Visão Computacional:** OpenCV, Pillow (para tratamento preliminar de imagens)

## 4. Instruções de Instalação e Execução
* Criar o ambiente virtual:
   ```bash
   python -m venv venv

1. Ativar o ambiente virtual:
No Windows: venv\Scripts\activate

2. Instalar as dependências:
pip install -r requirements.txt

3. Executar a API com Uvicorn:
uvicorn main:app --reload

5. Documentação da API
Interface Interativa (Swagger UI): Aceder a http://127.0.0.1:8000/docs após iniciar o servidor.

Endpoint Principal de Predição: /predict

6. Links das Evidências
Vídeo Explicativo (Google Drive): [Cole aqui o link do vídeo ao finalizar]

## 📊 Métricas e Desempenho do Modelo (Fase 1)

O modelo serializado em `models/modelo_mnist.pkl` é uma **Rede Neural Multi-Layer Perceptron (MLP)**. A tabela abaixo apresenta a comparação de desempenho obtida no conjunto de teste:

| Modelo | Acurácia Global | Precisão Ponderada | Recall Ponderado | F1-Score Ponderado |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 96.48% (0.9648) | 0.9648 | 0.9648 | 0.9648 |
| **SVM (RBF)** | 98.30% (0.9830) | 0.9830 | 0.9830 | 0.9830 |
| **Rede Neural MLP (Empregado na API)** | **97.89% (0.9789)** | **0.9789** | **0.9789** | **0.9789** |

*O modelo MLP foi o selecionado para servir a API devido ao excelente equilíbrio entre tempo de inferência, tamanho do artefato e alta capacidade de generalização.*