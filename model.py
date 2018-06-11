from preprocessing import load_data
from preprocessing import load_embeddings
from preprocessing import obtain_tfidf_values
from preprocessing import convert_words_into_embeddings

messages, classes = load_data('chats.json')
embeddings_dict, content = load_embeddings('SBW-vectors-300-min5.txt')
tfidf_values = obtain_tfidf_values(list(map(lambda msg_list: ' '.join(msg_list), messages)))
embeddings = convert_words_into_embeddings(messages, content, embeddings_dict)

messages_nueva = []
for i in range(0, len(messages)):
    """
        El primer elemento de la tupla es una lista de string (palabras)
        El segundo elemento es un np.array con los valores tfidf asociados
        El tercer elemento es una lista de np.array correspondiendo cada uno al embedding de cada palabra
    """
    messages_nueva.append((messages[i], tfidf_values[i].data, embeddings[i]))

# TODO: MULTIPLICAR CADA EMBEDDING POR EL TFIDF

"""
    Cada tupla debe tener: (lista_palabras, valores_tfidf, lista_embeddings)
    Cada tupla va a representar a una oración, o sea a un mensaje
    
    Luego para cada mensaje, a cada embedding, se lo multiplica por el tfidf asociado
    
    Cada mensaje va a estar representado por un embedding ponderado por cada palabra
    
    Por último promediar esos embeddings para obtener el vector final que represente a la oración    
"""


