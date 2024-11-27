# Reconhecimento de Sinais de Trânsito com Redes Neurais Convolucionais

Um projeto de machine learning para classificar sinais de trânsito usando uma rede neural convolucional (CNN), baseado no dataset German Traffic Sign Recognition Benchmark (GTSRB).

## 📖 Visão Geral do Projeto
Este projeto utiliza redes neurais convolucionais para reconhecer e classificar sinais de trânsito. A abordagem combina técnicas de pré-processamento de imagens e treinamento de uma CNN para identificar padrões e realizar a classificação de forma precisa.

 ## ⚙️ Instalação
Clone o repositório:
```
git clone https://github.com/samuelffreitas/IA_CNN_project.git
```

Acesse o diretório do projeto:
```
cd IA_CNN_project
```

Instale as dependências necessárias:

```
pip install -r requirements.txt
```

### Requisitos principais:

- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV

 ## 📊 Pré-Processamento
- Redimensionamento: As imagens são ajustadas para 48x48 pixels.
- Normalização: Os valores dos pixels são normalizados para o intervalo [0, 255], no formato float32.

## 🧠 Modelo
- Arquitetura: Rede Neural Convolucional (CNN) com 2 camadas convolucionais ( Convolução e Pooling ) e na rede neural densa de 2 camadas.
- Classificação: 43 categorias correspondentes aos diferentes sinais do dataset GTSRB, porém é utilizada para classificação apenas a do sinal de STOP, a número 14.
- Avaliação: A rede é avaliada tendo em conta a medida de acurácia.
  
## 📋 Resultados
- Acurácia no Conjunto de Teste ~= 90.0%.

A rede consegue identificar com sucesso os sinais de trânsito do tipo STOP.
