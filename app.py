from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import gdown  # <--- new import

app = Flask(__name__)

SELECTED_MODEL = 'CNN'  # KNN, CNN, CNNRF, YOLO, RL
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
EXTRACTOR_PATH = os.path.join(MODEL_DIR, 'mobilenetv2_extractor.h5')
KNN_PATH = os.path.join(MODEL_DIR, 'knn_model.joblib')
CNN_PATH = os.path.join(MODEL_DIR, 'modelo_cinto_otimizado.keras')
IMAGE_SIZE = (150, 150)
# Mapear classes para labels legíveis pela API
CLASS_MAP = {0.0: "com cinto", 1.0: "sem cinto"}

extractor = None
knn = None
cnn = None

def load_cnn_model():
    """Load CNN model, downloading from Google Drive if needed."""
    FILE_ID = "1DWJEK3o0xAdETxYExLxBGON4cJAleMlj"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    # Ensure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download if file does not exist
    if not os.path.exists(CNN_PATH):
        print(f"Downloading CNN model from Google Drive to {CNN_PATH}...")
        gdown.download(url, CNN_PATH, quiet=False)

    # Load the model
    cnn_model = load_model(CNN_PATH)
    return cnn_model

def load_knn_model():
    try:
        feature_extractor = load_model(EXTRACTOR_PATH, compile=False)
        knn_model = joblib.load(KNN_PATH)
        
        print("Modelos carregados com sucesso. O sistema está pronto para inferência.")
        return knn_model, feature_extractor

    except FileNotFoundError:
        print("\n--- ERRO CRÍTICO NA INFERÊNCIA ---")
        print(f"Verifique se ambos os arquivos existem:")
        print(f"1. Extrator Keras: {EXTRACTOR_PATH}")
        print(f"2. Classificador KNN: {KNN_PATH}")
        return None, None

def preprocess_image(img):
    # redimensiona a imagem para o tamanho esperado pelo extrator
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normaliza os pixels
    img_array = np.expand_dims(img_array, axis=0)
    # Usado pelo extrator MobileNetV2 (KNN branch)
    return preprocess_input(img_array)

def preprocess_image_for_cnn(img):
    """Pré-processamento compatível com o script classificar_m1.py:
    - redimensiona para IMAGE_SIZE
    - converte para array
    - normaliza dividindo por 255.0
    - adiciona dimensão de batch
    Retorna um array pronto para entrada no modelo CNN usado aqui.
    """
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img, knn_model, feature_extractor):
    processed_img = preprocess_image(img)
    features = feature_extractor.predict(processed_img, verbose=0)
    prediction_numeric = knn_model.predict(features)[0]
    predicted_class_name = CLASS_MAP.get(prediction_numeric, "CLASSE DESCONHECIDA")
    return predicted_class_name


@app.route('/api', methods=['GET'])
def check():
    return jsonify({"mensagem": "API online (GET)", "status": 200})


@app.route('/api', methods=['POST'])
def classify_image():
    if 'imagem' not in request.files:
        return jsonify({"mensagem": "Nenhum arquivo de imagem encontrado no campo 'imagem'", "status": 400}), 400

    file = request.files['imagem']

    if file.filename == '':
        return jsonify({"mensagem": "Nenhum arquivo selecionado", "status": 400}), 400

    img = Image.open(file.stream).convert('RGB')

    cnn = load_cnn_model()
    if cnn is not None:
        try:
            # Pré-processamento conforme classificar_m1.py (resize + /255.0)
            processed_img = preprocess_image_for_cnn(img)
            predicted = cnn.predict(processed_img)
            # Se o modelo retorna uma probabilidade (sigmoid) -> comparar com threshold 0.5
            pred_np = np.asarray(predicted)
            # caso seja probabilidade escalar ou shape (1,1)
            if pred_np.size == 1 or (pred_np.ndim == 2 and pred_np.shape[1] == 1):
                prob = float(np.squeeze(pred_np))
                # Seguir a lógica de classificar_m1.py: prob > 0.5 => "com cinto"
                predicted_label = CLASS_MAP.get(0.0) if prob > 0.5 else CLASS_MAP.get(1.0)
            # caso seja softmax com N classes (por exemplo (1,2)) -> usar argmax
            elif pred_np.ndim == 2 and pred_np.shape[1] > 1:
                class_idx = float(np.argmax(pred_np, axis=1)[0])
                predicted_label = CLASS_MAP.get(class_idx, "CLASSE DESCONHECIDA")
            else:
                # fallback genérico
                class_idx = float(np.round(np.squeeze(pred_np)))
                predicted_label = CLASS_MAP.get(class_idx, "CLASSE DESCONHECIDA")
            return jsonify({"mensagem": "Predição realizada", "predicao": predicted_label, "status": 200}), 200
        except Exception as e:
            return jsonify({"mensagem": "Erro ao processar a imagem para predição", "erro": str(e), "status": 500}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
