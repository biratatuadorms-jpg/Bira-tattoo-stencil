from flask import Flask, request, send_file
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Transformação de pré-processamento
preprocess = transforms.Compose([
    transforms.Resize((512,512)),  # ajusta resolução
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Função simples para gerar contorno (simulando Sketch Flow)
def generate_stencil(image_pil):
    img = preprocess(image_pil)
    img = img * 255
    img = img.squeeze(0).byte().numpy()
    
    # Aplicar contorno simples com Pillow / Numpy
    from PIL import ImageFilter, ImageOps
    pil_img = Image.fromarray(img)
    pil_img = pil_img.filter(ImageFilter.FIND_EDGES)  # detecção de borda
    pil_img = ImageOps.invert(pil_img)  # fundo branco, linhas pretas
    pil_img = pil_img.convert("L")
    return pil_img

@app.route('/stencil', methods=['POST'])
def stencil():
    file = request.files.get('image')
    if not file:
        return "Nenhuma imagem enviada", 400

    image_pil = Image.open(file.stream).convert("RGB")
    stencil_img = generate_stencil(image_pil)

    buf = io.BytesIO()
    stencil_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
