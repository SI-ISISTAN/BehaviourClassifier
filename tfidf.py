import json
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile

# Leo el archivo
with open('chats.json', encoding='utf8') as file:
    content = file.readlines()

# Cargo todos los chats en una lista
chats = []
for elem in content:
    current_line = json.loads(elem)
    chats.append(current_line.get('chats'))

# Como cada conversacion tiene una lista de chats, aplano y pongo todos los msjs en la misma lsita
flattened_chats = []
for chat in chats:
    for message in chat:
        flattened_chats.append(message)

# Convierto la lista a una que tenga sólo los mensajes que están etiquetados
flattened_chats = list(map(lambda x: x.get('text').lower(), filter(lambda x: len(x.keys()) > 5, flattened_chats)))

# Genero la matriz de TFIDF para los 32 documentos (oraciones) y 80 palabras en total -> Genera una matriz de (32, 80)
vectorizer = TfidfVectorizer()
flattened_chats = vectorizer.fit_transform(flattened_chats)

# Imprimo los valores para la primer oracion
print(flattened_chats[0])
