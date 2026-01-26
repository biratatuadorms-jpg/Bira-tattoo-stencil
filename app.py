import os
import gradio as gr
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

# ------------------------
# Carrega banco de tintas
# ------------------------
def carregar_tintas():
    df = pd.read_csv("tintas.csv")
    return df

tintas_df = carregar_tintas()

# ------------------------
# Distância de cor
# ------------------------
def distancia(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

# ------------------------
# Encontra a tinta mais próxima da marca escolhida
# ------------------------
def achar_tinta_mais_proxima(rgb, marca):
    banco = tintas_df[tintas_df["marca"] == marca]
    melhor = None
    menor_dist = 999999

    for _, row in banco.iterrows():
        cor_banco = (int(row["r"]), int(row["g"]), int(row["b"]))  # Garantir tipo int
        d = distancia(rgb, cor_banco)
        if d < menor_dist:
            menor_dist = d
            melhor = row

    return melhor

# ------------------------
# Mistura aproximada com primárias
# ------------------------
def mistura_primarias(rgb):
    r, g, b = rgb
    total = max(r+g+b, 1)

    pr = round((r/total)*100)
    pg = round((g/total)*100)
    pb = round((b/total)*100)

    return f"Vermelho: {pr}% | Amarelo: {pg}% | Azul: {pb}%"

# ------------------------
# Texto de aplicação no desenho
# ------------------------
def texto_aplicacao(rgb):
    r, g, b = rgb
    brilho = (r+g+b)/3

    if brilho > 200:
        return "Usar em áreas de luz, reflexos e pontos mais altos do volume."
    elif brilho > 120:
        return "Usar nas áreas de transição entre luz e sombra."
    elif brilho > 60:
        return "Usar para sombra média, base de profundidade."
    else:
        return "Usar para sombras profundas, recortes e áreas de maior peso visual."

# ------------------------
# Extrai TODAS as cores do desenho
# ------------------------
def extrair_cores(imagem, marca):
    img = np.array(imagem)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pixels = img.reshape((-1,3))
    pixels = np.float32(pixels)

    # K automático baseado na diversidade real do desenho
    k = min(25, len(np.unique(pixels, axis=0)))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    cores = np.uint8(kmeans.cluster_centers_)

    resultados = []

    for cor in cores:
        rgb = (int(cor[2]), int(cor[1]), int(cor[0]))
        tinta = achar_tinta_mais_proxima(rgb, marca)

        mistura = mistura_primarias(rgb)
        aplicacao = texto_aplicacao(rgb)

        resultados.append({
            "rgb": rgb,
            "nome": tinta["nome"],
            "mistura": mistura,
            "aplicacao": aplicacao
        })

    return resultados

# ------------------------
# Interface
# ------------------------
def processar(imagem, marca):
    resultados = extrair_cores(imagem, marca)

    html = "<div style='background:#111;padding:20px;'>"
    for r in resultados:
        cor = f"rgb{r['rgb']}"
        html += f"""
        <div style='background:#fff;margin:10px;padding:10px;border-radius:8px'>
            <div style='height:40px;background:{cor};border:1px solid #000'></div>
            <b>Nome da tinta:</b> {r['nome']}<br>
            <b>RGB:</b> {r['rgb']}<br>
            <b>Mistura primárias:</b> {r['mistura']}<br>
            <b>Onde aplicar:</b> {r['aplicacao']}
        </div>
        """
    html += "</div>"
    return html


with gr.Blocks(css="""
body { background:#000; }
h1, h2, p { color:black; }
""") as demo:

    gr.Markdown("# 🎨 Bira Tattoo – Paletas de Cores", elem_id="titulo_app", visible=True)
    gr.Markdown("Sistema profissional para extração de paletas reais baseadas na marca de tinta usada no estúdio.", elem_id="desc_app", visible=True)

    marca = gr.Dropdown(choices=tintas_df["marca"].unique().tolist(),
                         value="Electric Ink",
                         label="Selecione a marca da tinta")

    imagem = gr.Image(type="pil", label="Upload do desenho")

    btn = gr.Button("Gerar Paleta Profissional")

    saida = gr.HTML()

    btn.click(processar, inputs=[imagem, marca], outputs=saida)

# ------------------------
# Main guard para Render
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
