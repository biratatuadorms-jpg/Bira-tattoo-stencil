from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)  # <---- isso libera todas as requisições do seu HTML

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

    max_dim = 1024
    h, w = img.shape[:2]
    if max(h,w) > max_dim:
        scale = max_dim / max(h,w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    stencil_img = cv2.bitwise_not(edges)

    is_success, buffer = cv2.imencode(".png", stencil_img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
