import random
import re

# Dicionário de sinônimos e variações
synonyms = {
    "Olá": ["Oi", "Oie", "Salve", "Opa", "Eae", "Hey", "Oi oi"],
    "Oi": ["Olá", "Oie", "Salve", "Opa", "Eae"],
    "Tudo bem": ["tudo bom", "tudo blz", "tdb", "tudo ok", "como vai", "tudo certo"],
    "Você": ["vc", "tu", "vcs", "voce"],
    "você": ["vc", "tu", "voce"],
    "IA": ["AI", "bot", "robô", "inteligência", "sistema"],
    "inteligência artificial": ["IA", "AI", "sistema inteligente"],
    "Sim": ["s", "simm", "claro", "com certeza", "isso"],
    "Não": ["n", "nao", "nem", "negativo"],
    "Obrigado": ["vlw", "valeu", "obg", "agradecido"],
    "Tchau": ["falou", "até mais", "fui", "tchauzinho"],
    "Bom dia": ["dia", "bom dia!", "bom diaa"],
    "Boa tarde": ["tarde", "boa tardee"],
    "Boa noite": ["noite", "boa noitee"],
}

def augment_sentence(sentence):
    variations = [sentence]
    
    # 1. Substituição simples de sinônimos
    words = sentence.split()
    for i in range(len(words)):
        clean_word = words[i].replace("?", "").replace(".", "").replace(",", "")
        if clean_word in synonyms:
            for syn in synonyms[clean_word]:
                new_words = list(words)
                new_words[i] = syn + (words[i][len(clean_word):] if len(words[i]) > len(clean_word) else "")
                variations.append(" ".join(new_words))

    # 2. Variações de capitalização
    variations.append(sentence.lower())
    variations.append(sentence.upper())
    
    # 3. Remover pontuação
    variations.append(sentence.replace("?", "").replace(".", "").replace("!", ""))

    # 4. Repetição de letras finais (ex: Oiii)
    if len(sentence) > 3:
        variations.append(sentence[:-1] + sentence[-1]*3)

    return list(set(variations))

def main():
    with open('conversas_base.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    expanded_data = []
    
    for line in lines:
        if "<|user|>" in line and "<|assistant|>" in line:
            parts = line.split("<|assistant|>")
            user_part = parts[0].replace("<|user|>", "").strip()
            ia_part = parts[1].replace("<|end|>", "").strip()
            
            # Gera variações apenas para a pergunta do usuário
            user_variations = augment_sentence(user_part)
            
            for var in user_variations:
                expanded_data.append(f"<|user|> {var} <|assistant|> {ia_part} <|end|>\n")
    
    # Embaralha os dados para o modelo não decorar a ordem
    random.shuffle(expanded_data)
    
    # Salva o novo arquivo (v2 para manter a base limpa separada)
    with open('conversas_v2.txt', 'w', encoding='utf-8') as f:
        f.writelines(expanded_data)
    
    print(f"[SUCESSO] Dataset expandido de {len(lines)} para {len(expanded_data)} linhas no formato novo!")

if __name__ == "__main__":
    main()
