import os

input_file = 'conversas_base.txt'
output_file = 'conversas_base.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "Usuário:" in line and "IA:" in line:
        parts = line.split("IA:")
        user_part = parts[0].replace("Usuário:", "").strip()
        ia_part = parts[1].strip()
        new_lines.append(f"<|user|> {user_part} <|assistant|> {ia_part} <|end|>\n")
    elif line.strip():
        new_lines.append(line)

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Reformatadas {len(new_lines)} linhas.")
