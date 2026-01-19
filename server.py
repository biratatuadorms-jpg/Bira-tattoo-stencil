from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

def thinning(img):
    """Afinamento de linhas (Zhang-Suen) para stencil nítido"""
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

@app.route('/stencil', methods=['POST'])
def stencil():
    file = request.files.get('image')
    if not file:
        return "Nenhuma imagem enviada", 400

    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Redimensiona proporcionalmente se for muito grande
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Converter para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Filtro bilateral para suavizar mas manter contornos
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive Threshold para contornos consistentes
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Afinamento de linhas
    stencil_img = thinning(thresh)

    # Salvar em memória
    is_success, buffer = cv2.imencode(".png", stencil_img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
