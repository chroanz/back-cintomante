from flask import Flask, request, jsonify
import os
import sys
import logging
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import gdown

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
    FILE_ID = "1DWJEK3o0xAdETxYExLxBGON4cJAleMlj"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Verificando existência do arquivo de modelo em: %s", CNN_PATH)

    if not os.path.exists(CNN_PATH):
        logger.info("Arquivo do modelo não encontrado. Iniciando download de %s para %s", url, CNN_PATH)
        try:
            result = gdown.download(url, CNN_PATH, quiet=False)
            if result:
                logger.info("Download concluído: %s", CNN_PATH)
            else:
                logger.warning("gdown.download retornou None — verifique permissões/ID do arquivo.")
        except Exception:
            logger.exception("Falha ao baixar o modelo do Google Drive")

    try:
        logger.info("Carregando modelo CNN a partir de: %s", CNN_PATH)
        cnn_model = load_model(CNN_PATH)
        logger.info("Modelo CNN carregado com sucesso.")
        return cnn_model
    except Exception:
        logger.exception("Erro ao carregar o modelo CNN")
        raise

try:
    cnn = load_cnn_model()
except Exception:
    logger.error("Não foi possível carregar o modelo CNN na inicialização. A API continuará rodando, mas predições falharão.", exc_info=True)
    cnn = None

def preprocess_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normaliza os pixels
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def preprocess_image_for_cnn(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/api', methods=['GET'])
def check():
    logger.debug("GET /api chamada")
    return jsonify({"mensagem": "API online (GET)", "status": 200})

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

    if cnn is not None:
        try:
            processed_img = preprocess_image_for_cnn(img)
            logger.debug("Imagem pré-processada. Shape: %s dtype: %s", getattr(processed_img, 'shape', None), getattr(processed_img, 'dtype', None))
            predicted = cnn.predict(processed_img)
            logger.debug("Saída bruta do modelo: %s", repr(predicted))
            pred_np = np.asarray(predicted)

            if pred_np.size == 1 or (pred_np.ndim == 2 and pred_np.shape[1] == 1):
                prob = float(np.squeeze(pred_np))
                logger.info("Predição binária (probabilidade=%s)", prob)
                predicted_label = CLASS_MAP.get(0.0) if prob > 0.5 else CLASS_MAP.get(1.0)
            elif pred_np.ndim == 2 and pred_np.shape[1] > 1:
                class_idx = float(np.argmax(pred_np, axis=1)[0])
                logger.info("Predição multi-classe (idx=%s)", class_idx)
                predicted_label = CLASS_MAP.get(class_idx, "CLASSE DESCONHECIDA")
            else:
                class_idx = float(np.round(np.squeeze(pred_np)))
                logger.info("Predição rounding (idx=%s)", class_idx)
                predicted_label = CLASS_MAP.get(class_idx, "CLASSE DESCONHECIDA")

            logger.info("Predição final: %s", predicted_label)
            return jsonify({"mensagem": "Predição realizada", "predicao": predicted_label, "status": 200}), 200
        except Exception:
            logger.exception("Erro ao processar a imagem para predição")
            return jsonify({"mensagem": "Erro ao processar a imagem para predição", "erro": "ver logs do servidor", "status": 500}), 500
    else:
        logger.error("Modelo não carregado; impossível realizar predição")
        return jsonify({"mensagem": "Modelo não carregado", "status": 500}), 500

if __name__ == '__main__':
    logger.info("Iniciando Flask app na porta 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
