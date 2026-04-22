import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SemanticBrain:
    def __init__(self, memory_path="mente_da_ia.json", model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.memory_path = memory_path
        self.model = SentenceTransformer(model_name)
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_memory(self):
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)

    def aprender(self, frase):
        # Check if already learned to avoid redundancy
        for item in self.memory:
            if item['texto'].lower() == frase.lower():
                return False
        
        # Generate embedding
        vetor = self.model.encode(frase).tolist()
        self.memory.append({
            "texto": frase,
            "vetor": vetor
        })
        self._save_memory()
        return True

    def responder(self, pergunta, threshold=0.4):
        if not self.memory:
            return "Minha mente está vazia. Me ensine algo primeiro!", 0.0

        # Encode question
        vetor_pergunta = self.model.encode(pergunta)
        
        # Extract all stored vectors and ensure they are float32
        vetores_memoria = np.array([m['vetor'] for m in self.memory], dtype=np.float32)
        
        # Calculate cosine similarities
        similaridades = util.cos_sim(vetor_pergunta, vetores_memoria)[0]
        
        # Find the best match
        best_idx = np.argmax(similaridades)
        best_score = similaridades[best_idx].item()
        
        if best_score >= threshold:
            return self.memory[best_idx]['texto'], best_score
        else:
            return "Não tenho certeza sobre isso ainda...", best_score
