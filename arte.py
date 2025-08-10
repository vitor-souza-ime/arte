from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F

# Baixar modelo e processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Função para baixar imagem
def baixar_imagem(url):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

# Lista de classes/artistas
candidatos = [
    "Leonardo da Vinci",
    #"Rafael",
    #"Michelangelo",
    #"Rembrandt",
    #"Almeida Júnior",
    "Victor Meirelles",
    "Pedro Américo"
]

# Função de classificação zero-shot
def classificar_clip(url, candidatos):
    img = baixar_imagem(url)
    inputs = processor(text=candidatos, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_emb = model.get_image_features(**{k: inputs[k] for k in ["pixel_values"]})
        text_emb = model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
    # Normalização
    image_emb = F.normalize(image_emb, p=2, dim=-1)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    # Similaridades e probabilidades
    sims = (100.0 * image_emb @ text_emb.T).squeeze(0)
    probs = sims.softmax(dim=0)
    idx_max = torch.argmax(probs).item()
    return candidatos[idx_max], float(probs[idx_max])

# URLs de exemplo — todas obras em domínio público
obras = {
    "Mona Lisa": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
    "A Primeira Missa no Brasil": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Meirelles-primeiramissa2.jpg/800px-Meirelles-primeiramissa2.jpg",
    #"A Ronda Noturna": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/La_ronda_de_noche%2C_por_Rembrandt_van_Rijn.jpg/800px-La_ronda_de_noche%2C_por_Rembrandt_van_Rijn.jpg",
    #"Leitura": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Almeida_J%C3%BAnior_-_Leitura.jpg/800px-Almeida_J%C3%BAnior_-_Leitura.jpg",
    "Fala do Trono": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Pedro_Am%C3%A9rico_-_D._Pedro_II_na_abertura_da_Assembl%C3%A9ia_Geral.jpg/320px-Pedro_Am%C3%A9rico_-_D._Pedro_II_na_abertura_da_Assembl%C3%A9ia_Geral.jpg",
    
}

# Classificar cada obra
for nome_obra, url in obras.items():
    artista_previsto, prob = classificar_clip(url, candidatos)
    print(f"Obra: {nome_obra}")
    print(f"Artista previsto: {artista_previsto} ({prob*100:.2f}% de confiança)")
    print("-" * 60)
