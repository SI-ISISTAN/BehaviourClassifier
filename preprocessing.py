import json
import string


def load_data(json_dir):
    # Leo el archivo
    with open(json_dir, encoding='utf8') as file:
        content = file.readlines()

    # Cargo todos los chats en una lista
    msg_list = []
    for elem in content:
        current_line = json.loads(elem)
        for chat in current_line.get('chats'):
            msg_list.append(chat)

    # Armo tabla para mapear los signos de puntuación a None -> O sea para eliminarlos
    tabla_puntuacion = str.maketrans({key: None for key in string.punctuation})

    # Filtro los mensajes que están etiquetados, y después obtengo el texto del msj junto con la clase
    msg_list = list(filter(lambda x: 'IPA' in x.keys(), msg_list))
    msg_list = list(map(lambda x: (x.get('text').translate(tabla_puntuacion).split(), x.get('IPA')), msg_list))

    print('Se cargó el archivo:', json_dir)

    return msg_list


def load_embeddings(txt_dir):
    # Leo el archivo
    with open(txt_dir, encoding='utf8') as file:
        content = file.readlines()
    del content[0]  # La primer línea del TXT sólo indica las dimensiones de los embeddings

    print('Se cargó el archivo:', txt_dir)

    # Genero un diccionario cuya clave es el string con la palabra, y el valor el índice de la palabra buscada
    embedding_dict = {}
    id_embedding = 0
    for word in list(map(lambda x: x.split()[0], content)):
        embedding_dict[word] = id_embedding
        id_embedding += 1

    print('Se generó el índice de embeddings')

    # Retorno el diccionario, y la lista de líneas que representa el TXT
    return embedding_dict, content


def format_embedding(raw_line):
    aux = raw_line.split()
    del aux[0]  # Elimino el primer elemento que se corresponde con la primer palabra de la línea

    return list(map(lambda x: float(x), aux))
