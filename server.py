from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

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

    # Redimensiona proporcionalmente se muito grande
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Converter para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Suavização para remover ruído sem borrar contornos
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive Threshold para contornos consistentes
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)  # Fundo branco, linhas pretas

    # Dilatar levemente para linhas contínuas
    kernel = np.ones((2,2), np.uint8)
    stencil_img = cv2.dilate(thresh, kernel, iterations=1)

    # Salvar em memória
    is_success, buffer = cv2.imencode(".png", stencil_img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
