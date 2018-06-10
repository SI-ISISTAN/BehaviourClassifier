from preprocessing import load_data
from preprocessing import load_embeddings


msg_list = load_data('chats.json')
embeddings_dict, content = load_embeddings('SBW-vectors-300-min5.txt')  # Tarda un par de minutos en leerse el contenido

# Aplicar stemming porque sino tira KeyError
msg_list_nueva = []
for msg in msg_list:
    words = msg[0]
    ipa = msg[1]
    words = list(map(lambda word: content[embeddings_dict[word]], words))
    msg_list_nueva.append((words, ipa))

# msg_list = list(map(lambda x: (content[embeddings_dict[x[0]]], x[1]), msg_list))  # Convierto cada palabra a embedding

# print(content[embeddings_dict['casa']])
