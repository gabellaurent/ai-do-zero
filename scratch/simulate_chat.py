
import torch
import sys
import os
sys.path.append(os.getcwd())
from model import MiniTransformer, device

# Carregar o modelo e metadados
checkpoint = torch.load('slm_model.pth', map_location=device)
chars = checkpoint['chars']
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = len(chars)

model = MiniTransformer(vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

def get_response(prompt):
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        # Usando os novos parâmetros
        generated_tokens = model.generate(context, max_new_tokens=50, temperature=0.7, top_k=5)[0].tolist()
    full_text = decode(generated_tokens)
    new_content = full_text[len(prompt):]
    response = new_content.split("\n")[0].split("Usuário:")[0].strip()
    return response

print("--- Simulação de Chat Melhorado ---")
historico = []
test_inputs = [
    "Oi",
    "Sim, e você?",
    "Como você está?",
    "Estou bem.",
    "Qual seu nome?",
    "O meu também!",
    "Legal, vc mora no Brasil?",
    "KKKK",
    "Muito bom"
]

for inp in test_inputs:
    contexto_recente = "\n".join(historico[-3:])
    if contexto_recente:
        prompt = f"{contexto_recente}\nUsuário: {inp} IA:"
    else:
        prompt = f"Usuário: {inp} IA:"
    
    resp = get_response(prompt)
    print(f"Usuário: {inp}")
    print(f"IA: {resp}")
    historico.append(f"Usuário: {inp} IA: {resp}")
