import torch
import os
from model import MiniTransformer, device, block_size, batch_size

# Configurações de treino
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

# 1. Carregar o dataset
with open('conversas_v2.txt', 'r', encoding='utf-8') as f:
    text = f.read()

checkpoint_path = 'slm_model.pth'
start_iter = 0

# 2. Inicializar Modelo e Vocabulário
if os.path.exists(checkpoint_path):
    print(f"\n[INFO] Carregando checkpoint de '{checkpoint_path}'...")
    # Carrega no dispositivo correto
    checkpoint = torch.load(checkpoint_path, map_location=device)
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = len(chars)
    
    model = MiniTransformer(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_iter = checkpoint.get('iter', 0)
    print(f"[INFO] Retomando treinamento do passo {start_iter}")
else:
    print("\n[INFO] Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    model = MiniTransformer(vocab_size)

model.to(device)

# Mapeamentos (usando if c in stoi para evitar erro com caracteres novos no arquivo)
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# 3. Preparar dados
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% para treino, resto para validação
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Print o número de parâmetros
print(f"{sum(p.numel() for p in model.parameters())/1e6}M parâmetros")

# Criar um otimizador
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Função para salvar o checkpoint
def save_model(model, iter_count, loss_val):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'chars': chars,
        'stoi': stoi,
        'itos': itos,
        'iter': iter_count,
        'loss': loss_val
    }
    torch.save(checkpoint, 'slm_model.pth')
    print(f"\n[SALVO] Modelo salvo em 'slm_model.pth' (Passo {iter_count}, Loss {loss_val:.4f})")

# 4. Loop de Treinamento
print("Iniciando treinamento... (Pressione Ctrl+C para parar e salvar o progresso atual)")
try:
    # Começa do start_iter e vai até max_iters + start_iter para garantir que treine o total solicitado
    for iter in range(start_iter, start_iter + max_iters):
        
        # a cada eval_interval, avalia a perda e salva o progresso
        if iter % eval_interval == 0 or iter == (start_iter + max_iters - 1):
            losses = estimate_loss()
            print(f"passo {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            save_model(model, iter, losses['train'])

        # sorteia um lote de dados
        xb, yb = get_batch('train')

        # avalia a perda
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

except KeyboardInterrupt:
    print("\nTreinamento interrompido pelo usuário.")
finally:
    # Tenta salvar o estado atual se iter existir
    try:
        save_model(model, iter, loss.item())
    except:
        pass
    print("Treinamento finalizado.")
