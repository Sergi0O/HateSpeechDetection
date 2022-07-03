import re

# URLs
URLS_RE = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")
# Caracteres no alfanuméricos y dígitos
SYMBOL_RE = re.compile(r"\W+|\d+")
# Menciones de usuarios
USER_RE = re.compile(r"@\w+")
# Hashtags
HASHTAG_RE = re.compile(r"#(\w+)")
# d -> de
DE_RE = re.compile(r" d |^d | d$")
# (q , k) -> que
QUE_RE = re.compile(r" q |^q | q$| k |^k | k$")
# t -> te
TE_RE = re.compile(r" t |^t | t$")


def split_hashtag(match):
    s = match.group(1)
    words = re.split(r'(?=[A-Z])', s)
    sentence = ""
    for i in range(len(words) - 1):
        sentence += words[i]
        if len(words[i]) > 1 or len(words[i+1]) > 1:
            sentence += " "

    if len(words):
        sentence += words[-1]

    return sentence


def preprocess(text):
    text = HASHTAG_RE.sub(split_hashtag, text)      # Separa las palabras de los hashtags
    text = USER_RE.sub('', text)                    # Elimina los @usuarios
    text = URLS_RE.sub('', text)                    # Elimina URLs
    text = SYMBOL_RE.sub(' ', text)                 # Elimina caracteres no alfanuméricos y dígitos
    text = text.lower()                             # Minúscula
    text = DE_RE.sub(' de ', text)                  # Sustituye d por de
    text = QUE_RE.sub(' que ', text)                # Sustituye q o k por que
    text = TE_RE.sub(' te ', text)                  # Sustituye t por te
    text = ' '.join(text.split())                   # Elimina varios espacios

    return text
