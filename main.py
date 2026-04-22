import sys
from brain import SemanticBrain

def main():
    print("--- Tabula Rasa AI Initialized ---")
    print("Commands: /ensinar <frase>, /perguntar <pergunta>, /sair")
    
    brain = SemanticBrain()
    
    while True:
        try:
            user_input = input("\nVocê: ").strip()
            
            if user_input.lower() == "/sair":
                print("Até logo!")
                break
                
            if user_input.startswith("/ensinar "):
                frase = user_input[9:]
                if brain.aprender(frase):
                    print(f"IA: Entendido. Agora eu acredito que: '{frase}'")
                else:
                    print("IA: Eu já sabia disso!")
                    
            elif user_input.startswith("/perguntar "):
                pergunta = user_input[11:]
                resposta, score = brain.responder(pergunta)
                print(f"IA (Confiança: {score:.2f}): {resposta}")
                
            else:
                print("IA: Use /ensinar para me dar uma crença ou /perguntar para testar minha memória.")
                
        except KeyboardInterrupt:
            print("\nSaindo...")
            break

if __name__ == "__main__":
    main()
