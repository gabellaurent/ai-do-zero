
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
        generated_tokens = model.generate(context, max_new_tokens=50)[0].tolist()
    full_text = decode(generated_tokens)
    new_content = full_text[len(prompt):]
    response = new_content.split("\n")[0].split("Usuário:")[0].strip()
    return response

print("--- Teste de Memória ---")
h = []

# Turno 1
u1 = "Olá, meu nome é Carlos."
prompt1 = f"Usuário: {u1} IA:"
r1 = get_response(prompt1)
print(f"U: {u1}")
print(f"IA: {r1}")
h.append(f"Usuário: {u1} IA: {r1}")

# Turno 2 (Referência ao nome)
u2 = "Qual o meu nome?"
prompt2 = "\n".join(h) + f"\nUsuário: {u2} IA:"
r2 = get_response(prompt2)
print(f"U: {u2}")
print(f"IA: {r2}")
