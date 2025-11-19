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
EXTRACTOR_PATH = os.path.join(MODEL_DIR, 'mobilenetv2_extractor.h5') 
KNN_PATH = os.path.join(MODEL_DIR, 'knn_model.joblib') 
CNN_PATH = os.path.join(MODEL_DIR, 'modelo_cinto_otimizado.keras') 
IMAGE_SIZE = (150, 150)
# Mapear classes para labels legíveis pela API
CLASS_MAP = {0.0: "com cinto", 1.0: "sem cinto"}

extractor = None
knn = None
cnn = None 

KNN_FILE_ID = os.getenv("KNN_FILE_ID", "1TYZYKLoCS2D8Ks0hK-_R_p28RCuTQO0N") 
EXTRACTOR_FILE_ID = os.getenv("EXTRACTOR_FILE_ID", "1zyDpaJSwUvvj3b8GZ5tvt-4xIhr_8VS2")

def load_model_with_fallback(file_path, file_id, loader_func, logger):
    url = f"https://drive.google.com/uc?id={file_id}"
    model = None
    
    # Tentar carregar localmente se o arquivo já existir no repo/slug
    if os.path.exists(file_path):
        try:
            logger.info("Carregando modelo a partir de arquivo no disco: %s", file_path)
            model = loader_func(file_path)
            logger.info("Modelo carregado com sucesso a partir do disco.")
            return model
        except Exception:
            logger.exception(f"Falha ao carregar o modelo a partir do arquivo no disco: {file_path}. Tentando download...")

    # Caso contrário, baixe para um diretório temporário, carregue em memória e remova os arquivos temporários.
    if file_id and file_id not in ["ID_PLACEHOLDER_KNN_JOBLIB", "ID_PLACEHOLDER_EXTRACTOR_H5", None]:
        logger.info("Arquivo do modelo não encontrado localmente. Tentando download temporário de: %s", url)
        try:
            with tempfile.TemporaryDirectory() as td:
                temp_path = os.path.join(td, os.path.basename(file_path))
                
                result = gdown.download(url, temp_path, quiet=False)
                if not result or not os.path.exists(temp_path):
                    logger.warning("gdown.download falhou ou não criou o arquivo esperado: %s", temp_path)
                    return None
                
                logger.info("Download temporário concluído: %s", temp_path)

                load_target = temp_path
                
                # Caso exista um zip, extraímos automaticamente
                if zipfile.is_zipfile(temp_path) or temp_path.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(temp_path, 'r') as z:
                            z.extractall(td)
                        logger.info("Zip extraído em tempdir: %s", td)
                        
                        candidate = None
                        for root, _, files in os.walk(td):
                            for f in files:
                                if f == os.path.basename(file_path) or f.lower().endswith(('.keras', '.h5', '.joblib', '.pkl')):
                                    candidate = os.path.join(root, f)
                                    break
                            if candidate:
                                break
                        load_target = candidate if candidate else temp_path

                    except Exception:
                        logger.exception("Falha ao extrair zip temporário: %s", temp_path)

                try:
                    logger.info("Carregando modelo a partir de recurso temporário: %s", load_target)
                    model = loader_func(load_target)
                    logger.info("Modelo carregado em memória com sucesso (temporário).")
                    return model
                except Exception:
                    logger.exception(f"Falha ao carregar o modelo a partir do recurso temporário: {load_target}")
        except Exception:
            logger.exception("Falha durante download/carregamento temporário do modelo")
    
    return None

def load_knn_mobilenet_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    def load_keras_model(path):
        return load_model(path, compile=False)

    def load_joblib_model(path):
        return joblib.load(path)

    extractor_model = load_model_with_fallback(
        file_path=EXTRACTOR_PATH,
        file_id=EXTRACTOR_FILE_ID,
        loader_func=load_keras_model,
        logger=logger
    )
    
    knn_model = load_model_with_fallback(
        file_path=KNN_PATH,
        file_id=KNN_FILE_ID,
        loader_func=load_joblib_model,
        logger=logger
    )
    
    if extractor_model is None or knn_model is None:
        raise FileNotFoundError("Não foi possível carregar ou baixar ambos os modelos necessários (Extrator MobileNetV2 e KNN Classifier).")
        
    return knn_model, extractor_model

try:
    knn, extractor = load_knn_mobilenet_models()
    logger.info("Sistema de predição (KNN + MobileNetV2) carregado com sucesso.")
except Exception:
    logger.error("Não foi possível carregar os modelos KNN/MobileNetV2 na inicialização. A API continuará rodando, mas predições falharão.", exc_info=True)
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
    return jsonify({"mensagem": "API online (GET)", "status": 200, "modelo": SELECTED_MODEL, "status_modelo": model_status}), 200

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