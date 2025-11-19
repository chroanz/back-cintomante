from flask import Flask, request, jsonify
import os
import sys
import logging
import tempfile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.preprocessing import image
import joblib
import gdown
import zipfile

# Logging configuration
logger = logging.getLogger("seatbelt_api")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

app = Flask(__name__)
# integrate flask logger handlers (optional)
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

SELECTED_MODEL = 'KNN'  
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
EXTRACTOR_PATH = os.path.join(MODEL_DIR, 'mobilenetv2_extractor.h5') # O seu extrator MobileNetV2
KNN_PATH = os.path.join(MODEL_DIR, 'knn_model.joblib') # O seu modelo KNN
CNN_PATH = os.path.join(MODEL_DIR, 'modelo_cinto_otimizado.keras') 
IMAGE_SIZE = (150, 150)
# Mapear classes para labels legíveis pela API
CLASS_MAP = {0.0: "com cinto", 1.0: "sem cinto"}

extractor = None
knn = None
cnn = None 

def load_knn_mobilenet_models():
    logger.info("Carregando o Extrator de Características (MobileNetV2) a partir de: %s", EXTRACTOR_PATH)
    try:
        if os.path.exists(EXTRACTOR_PATH):
            extractor_model = load_model(EXTRACTOR_PATH, compile=False)
            logger.info("Extrator carregado com sucesso a partir do disco.")
        else:
            raise FileNotFoundError(f"Arquivo do extrator não encontrado: {EXTRACTOR_PATH}")
    except Exception:
        logger.exception("Falha ao carregar o Extrator MobileNetV2.")
        extractor_model = None

    logger.info("Carregando o Classificador KNN a partir de: %s", KNN_PATH)
    try:
        if os.path.exists(KNN_PATH):
            knn_model = joblib.load(KNN_PATH)
            logger.info("Modelo KNN carregado com sucesso.")
        else:
            raise FileNotFoundError(f"Arquivo do KNN não encontrado: {KNN_PATH}")
    except Exception:
        logger.exception("Falha ao carregar o modelo KNN.")
        knn_model = None

    if extractor_model is None or knn_model is None:
        raise FileNotFoundError("Não foi possível carregar ambos os modelos necessários (Extrator e KNN).")

    return knn_model, extractor_model

try:
    knn, extractor = load_knn_mobilenet_models()
    logger.info("Sistema de predição (KNN + MobileNetV2) carregado com sucesso.")
except Exception:
    logger.error("Não foi possível carregar os modelos KNN/MobileNetV2 na inicialização.", exc_info=True)
    knn = None
    extractor = None

def preprocess_image_for_knn(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return mobilenet_preprocess_input(img_array)

@app.route('/api', methods=['GET'])
def check():
    logger.debug("GET /api chamada")
    model_status = "ok" if knn is not None and extractor is not None else "erro de carregamento"
    return jsonify({"mensagem": "API online (GET)", "status": 200, "modelo": SELECTED_MODEL, "status_modelo": model_status})

@app.route('/api', methods=['POST'])
def classify_image():
    logger.info("POST /api recebida de %s", request.remote_addr)
    if 'imagem' not in request.files:
        logger.warning("Campo 'imagem' ausente na requisição")
        return jsonify({"mensagem": "Nenhum arquivo de imagem encontrado no campo 'imagem'", "status": 400}), 400

    file = request.files['imagem']
    logger.info("Arquivo recebido: filename=%s content_type=%s", file.filename, file.content_type)

    if file.filename == '':
        logger.warning("Nenhum arquivo selecionado (filename vazio)")
        return jsonify({"mensagem": "Nenhum arquivo selecionado", "status": 400}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        logger.debug("Imagem aberta com sucesso. Tamanho original: %s", img.size)
    except Exception:
        logger.exception("Erro ao abrir a imagem enviada")
        return jsonify({"mensagem": "Erro ao abrir a imagem enviada", "status": 400}), 400

    if knn is not None and extractor is not None:
        try:
            processed_img = preprocess_image_for_knn(img)
            logger.debug("Imagem pré-processada. Shape: %s dtype: %s", processed_img.shape, processed_img.dtype)

            features = extractor.predict(processed_img, verbose=0)
            logger.debug("Características extraídas. Shape: %s", features.shape)
            
            prediction_numeric = knn.predict(features)[0] 
            
            logger.info("Predição numérica bruta (KNN): %s", prediction_numeric)
            
            class_idx = float(prediction_numeric)
            predicted_label = CLASS_MAP.get(class_idx, "CLASSE DESCONHECIDA")

            logger.info("Predição final: %s (Modelo: %s)", predicted_label, SELECTED_MODEL)
            return jsonify({"mensagem": "Predição realizada", "predicao": predicted_label, "modelo": SELECTED_MODEL, "status": 200}), 200
        except Exception:
            logger.exception("Erro ao processar a imagem para predição (KNN + MobileNetV2)")
            return jsonify({"mensagem": "Erro ao processar a imagem para predição", "erro": "ver logs do servidor", "status": 500}), 500
    else:
        logger.error("Modelos (KNN e/ou Extrator) não carregados; impossível realizar predição")
        return jsonify({"mensagem": "Modelos não carregados", "modelo": SELECTED_MODEL, "status": 500}), 500

if __name__ == '__main__':
    logger.info("Iniciando Flask app na porta 5000 (Modelo: %s)", SELECTED_MODEL)
    app.run(host='0.0.0.0', port=5000, debug=False)