import re

VOCABULARY_PUNCTUATION = ['!', '?', '.', ',']

unique = list()
with open('data/fra.txt', "r", encoding='utf-8') as f:
    lines = f.read().split("\n")
    for line in lines:
        input_text, _, _ = line.split("\t")
        if input_text in unique:
            continue

        unique.append(input_text)

with open('data/pnc-eng.txt', 'w', encoding='utf-8') as f:
    for unique_text in unique:
        f.write(''.join(re.split("[" + "\\".join(VOCABULARY_PUNCTUATION) + "]", unique_text)).lower())
        f.write('\t')
        f.write(unique_text.lower())
        f.write('\n')
