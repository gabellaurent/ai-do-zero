import torch
import torch.optim as optim
from model import MiniTransformer, device, block_size

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

# Otimizador para aprendizado em tempo real
optimizer = optim.AdamW(model.parameters(), lr=5e-5) # Taxa reduzida para estabilidade

def aprender(texto_interacao):
    """ Realiza um mini-treino com a nova interação """
    model.train()
    # Garante que o texto não ultrapasse o block_size do modelo
    if len(texto_interacao) > block_size:
        texto_interacao = texto_interacao[-block_size:]
        
    tokens = torch.tensor([encode(texto_interacao)], dtype=torch.long, device=device)
    if tokens.shape[1] < 2: return # Precisa de pelo menos 2 tokens para prever o próximo
    
    # X são os tokens de entrada, Y são os mesmos tokens deslocados (o alvo)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.eval()

def salvar_progresso():
    """ Salva os pesos atualizados no arquivo .pth """
    checkpoint['model_state_dict'] = model.state_dict()
    torch.save(checkpoint, 'slm_model.pth')
    print("\n[SISTEMA] Conhecimento salvo com sucesso!")

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
        contexto_recente = "".join(historico[-5:]) # Aumentado para 5 trocas
        if contexto_recente:
            prompt = f"{contexto_recente}<|user|> {user_input} <|assistant|>"
        else:
            prompt = f"<|user|> {user_input} <|assistant|>"
            
        # Garante que o prompt não ultrapasse o block_size (limita caracteres aproximados)
        if len(prompt) > block_size: 
            prompt = prompt[-block_size:]
            
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        
        # Gera a resposta com temperatura mais baixa para ser mais precisa
        with torch.no_grad():
            generated_tokens = model.generate(context, max_new_tokens=100, temperature=0.7, top_k=5)[0].tolist()
        
        # Decodifica
        full_text = decode(generated_tokens)
        
        # Extrai apenas a nova resposta da IA
        # Pega o que vem depois do último "IA:" do prompt
        new_content = full_text[len(prompt):]
        # Para no primeiro newline ou na próxima tag de Usuário ou IA
        response = new_content.split("<|end|>")[0].split("<|user|>")[0].split("<|assistant|>")[0].strip()
        
        # Se a geração parou no meio de um token especial, remove os caracteres residuais
        for token in ["<|", "<|u", "<|us", "<|use", "<|user", "<|user|", "<|a", "<|as", "<|ass", "<|assi", "<|assis", "<|assist", "<|assista", "<|assistan", "<|assistant", "<|assistant|", "<|e", "<|en", "<|end"]:
            if response.endswith(token):
                response = response[:-len(token)].strip()
            
        print(f"IA: {response}")
        
        # Adiciona ao histórico no novo formato
        interacao_completa = f"<|user|> {user_input} <|assistant|> {response} <|end|>"
        historico.append(interacao_completa)
        
        # Aprendizado Ativo: Aprende com a interação agora mesmo
        aprender(interacao_completa)
        
        # Salva a nova conversa no dataset para treinos futuros
        with open('conversas_base.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n{interacao_completa}")

except KeyboardInterrupt:
    print("\nIA: Tchau! Conversamos mais depois.")
finally:
    salvar_progresso()
