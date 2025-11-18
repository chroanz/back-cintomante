from flask import Flask, request, jsonify
import os
import sys
import logging
import tempfile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
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
    FILE_ID = os.getenv("MODEL_FILE_ID", "1DWJEK3o0xAdETxYExLxBGON4cJAleMlj")
    url = os.getenv("MODEL_URL", f"https://drive.google.com/uc?id={FILE_ID}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Verificando existência do arquivo de modelo em: %s", CNN_PATH)
    # Caso exista um zip do modelo no repositório (ex: modelo_cinto_otimizado.keras.zip
    # ou modelo_cinto_otimizado.zip) extraímos automaticamente para o diretório de modelos.
    possible_zip_names = [CNN_PATH + '.zip',
                          os.path.join(MODEL_DIR, os.path.basename(CNN_PATH) + '.zip'),
                          os.path.join(MODEL_DIR, os.path.splitext(os.path.basename(CNN_PATH))[0] + '.zip')]
    for zip_path in possible_zip_names:
        if os.path.exists(zip_path):
            logger.info("Arquivo zip do modelo encontrado: %s. Extraindo...", zip_path)
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(MODEL_DIR)
                logger.info("Extração concluída: %s", zip_path)
            except Exception:
                logger.exception("Falha ao extrair o arquivo zip do modelo: %s", zip_path)

    # Tentar carregar localmente se o arquivo já existir no repo/slug
    if os.path.exists(CNN_PATH):
        try:
            logger.info("Carregando modelo CNN a partir de arquivo no disco: %s", CNN_PATH)
            cnn_model = load_model(CNN_PATH)
            logger.info("Modelo CNN carregado com sucesso a partir do disco.")
            return cnn_model
        except Exception:
            logger.exception("Falha ao carregar o modelo a partir do arquivo no disco: %s", CNN_PATH)

    # Caso contrário, baixe para um diretório temporário, carregue em memória e remova os arquivos temporários.
    logger.info("Arquivo do modelo não encontrado localmente. Tentando download temporário de: %s", url)
    try:
        with tempfile.TemporaryDirectory() as td:
            temp_path = os.path.join(td, os.path.basename(CNN_PATH))
            result = gdown.download(url, temp_path, quiet=False)
            if not result or not os.path.exists(temp_path):
                logger.warning("gdown.download falhou ou não criou o arquivo esperado: %s", temp_path)
            else:
                logger.info("Download temporário concluído: %s", temp_path)

                # Se for zip, extraia em tempdir
                if zipfile.is_zipfile(temp_path) or temp_path.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(temp_path, 'r') as z:
                            z.extractall(td)
                        logger.info("Zip extraído em tempdir: %s", td)
                    except Exception:
                        logger.exception("Falha ao extrair zip temporário: %s", temp_path)

                # Procurar por um arquivo .keras ou um diretório SavedModel dentro do tempdir
                candidate = None
                for root, _, files in os.walk(td):
                    for f in files:
                        if f.lower().endswith('.keras'):
                            candidate = os.path.join(root, f)
                            break
                    if candidate:
                        break

                saved_model_dir = None
                for name in os.listdir(td):
                    p = os.path.join(td, name)
                    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'saved_model.pb')):
                        saved_model_dir = p
                        break

                load_target = saved_model_dir or candidate or temp_path
                try:
                    logger.info("Carregando modelo a partir de recurso temporário: %s", load_target)
                    cnn_model = load_model(load_target)
                    logger.info("Modelo carregado em memória com sucesso (temporário).")
                    return cnn_model
                except Exception:
                    logger.exception("Falha ao carregar o modelo a partir do recurso temporário: %s", load_target)
    except Exception:
        logger.exception("Falha durante download/carregamento temporário do modelo")

    # Se tudo falhar, levantar exceção para ser tratada pelo chamador
    raise FileNotFoundError(f"Não foi possível obter um modelo válido a partir de {CNN_PATH} ou {url}")

try:
    cnn = load_cnn_model()
except Exception:
    logger.error("Não foi possível carregar o modelo CNN na inicialização. A API continuará rodando, mas predições falharão.", exc_info=True)
    cnn = None

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
