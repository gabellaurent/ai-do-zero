
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

test_cases = [
    "Olá",
    "Oi",
    "oie",
    "Tudo bem?",
    "Quem é você?",
    "Qual o seu nome?",
    "vc eh um robo?",
    "Como vai vc?",
    "O que vc faz?",
    "Me conte uma piada",
    "Quem te criou?",
    "O que voce gosta de comer?",
    "Tchau"
]

print(f"{'Usuário':<30} | {'IA'}")
print("-" * 60)

for user_input in test_cases:
    prompt = f"Usuário: {user_input} IA:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=50)[0].tolist()
    
    full_text = decode(generated_tokens)
    try:
        response = full_text.split("IA:")[1].split("\n")[0].strip()
    except:
        response = full_text[len(prompt):].split("\n")[0].strip()
    
    print(f"{user_input:<30} | {response}")
