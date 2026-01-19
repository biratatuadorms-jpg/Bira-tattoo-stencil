from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # Permite requisições do HTML

@app.route('/stencil', methods=['POST'])
def stencil():
    file = request.files.get('image')
    if not file:
        return "Nenhuma imagem enviada", 400

    # Ler imagem enviada
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Redimensiona se muito grande
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Converter para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para suavizar mas manter bordas
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Detecção de borda Canny mais agressiva
    edges = cv2.Canny(gray, threshold1=40, threshold2=120)

    # Dilatar as linhas para ficarem mais grossas e contínuas
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Inverter: fundo branco, linhas pretas
    stencil_img = cv2.bitwise_not(edges)

    # Salvar em memória
    is_success, buffer = cv2.imencode(".png", stencil_img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
