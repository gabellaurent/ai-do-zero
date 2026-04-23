import torch
from model import MiniTransformer, device

# Carregar o modelo e metadados
if not torch.cuda.is_available():
    checkpoint = torch.load('slm_model.pth', map_location=torch.device('cpu'))
else:
    checkpoint = torch.load('slm_model.pth')

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

print("--- SLM Chat Iniciado ---")
print("Digite 'sair' para encerrar.")

historico = []

try:
    while True:
        user_input = input("Usuário: ")
        if user_input.lower() == 'sair':
            print("IA: Até logo!")
            break
        
        # Mantém apenas as últimas 3 trocas para não estourar o contexto (block_size=64)
        contexto_recente = "\n".join(historico[-3:])
        if contexto_recente:
            prompt = f"{contexto_recente}\nUsuário: {user_input} IA:"
        else:
            prompt = f"Usuário: {user_input} IA:"
            
        # Garante que o prompt não ultrapasse o block_size (limita caracteres aproximados)
        if len(prompt) > 200: 
            prompt = prompt[-200:]
            
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        
        # Gera a resposta com temperatura mais baixa para ser mais precisa
        with torch.no_grad():
            generated_tokens = model.generate(context, max_new_tokens=50, temperature=0.7, top_k=5)[0].tolist()
        
        # Decodifica
        full_text = decode(generated_tokens)
        
        # Extrai apenas a nova resposta da IA
        # Pega o que vem depois do último "IA:" do prompt
        new_content = full_text[len(prompt):]
        # Para no primeiro newline ou na próxima tag
        response = new_content.split("\n")[0].split("Usuário:")[0].strip()
            
        print(f"IA: {response}")
        
        # Adiciona ao histórico
        historico.append(f"Usuário: {user_input} IA: {response}")

except KeyboardInterrupt:
    print("\nIA: Tchau! Conversamos mais depois.")
