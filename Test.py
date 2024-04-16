vocab = ["Hej", "den", "Det", "test"]
vocab = list(map(lambda _str: _str.lower(), vocab))
special_tokens = ["UNK"]

import re
token_pattern = re.compile(r'|'.join(sorted(vocab, key=len, reverse=True)), re.UNICODE | re.IGNORECASE)
vocab = special_tokens + vocab

word2indices = {word: i for i, word in enumerate(vocab)}
indices2word = {i: word for i, word in enumerate(vocab)}

import re
msg = "Hej, det här är ett test"
print("Message:", msg)

print("Word indices:", end=" ")

idxs = []
for word in re.findall(r'\w+', msg):
    idxs.append(word2indices.get(word.lower(), word2indices["UNK"]))
print(idxs)

all_matches = list(token_pattern.finditer(msg))
pass