import json
import string
import numpy as np

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(json_dir):
    def delete_punctuation(words_list):
        # Armo tabla para mapear los signos de puntuación a None -> O sea para eliminarlos
        tabla_puntuacion = str.maketrans({key: None for key in string.punctuation})
        return list(map(lambda x: x.translate(tabla_puntuacion), words_list))

    def stem_words(words_list):
        # Armo un stemmer para reducir las palabras a su raíz y se lo aplico a la lista
        stemmer = SnowballStemmer('spanish')
        return list(map(lambda x: stemmer.stem(x), words_list))

    # Leo el archivo
    with open(json_dir, encoding='utf8') as file:
        content = file.readlines()

    # Cargo todos los chats en una lista
    msg_list = []
    for elem in content:
        current_line = json.loads(elem)
        for chat in current_line.get('chats'):
            msg_list.append(chat)

    # Filtro los mensajes que están etiquetados
    msg_list = list(filter(lambda x: 'IPA' in x.keys(), msg_list))

    # A partir del texto, elimino signos de puntuación y aplico stemming
    # messages = list(map(lambda x: stem_words(delete_punctuation(x.get('text').split())), msg_list))
    messages = list(map(lambda x: delete_punctuation(x.get('text').split()), msg_list))

    # Genero un array con las clases correspondientes a cada mensaje
    classes = list(map(lambda x: x.get('IPA'), msg_list))
    classes = np.array(list(map(lambda x: x if x is not None else '-1', classes)))
    classes = classes.astype(np.int)

    print('Se cargó el archivo:', json_dir)

    return messages, classes


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


def replace_word(rw_content, rw_embeddings_dict, rw_word):
        try:
            embedding = format_embedding_line(rw_content[rw_embeddings_dict[rw_word]])
        except KeyError:
            embedding = np.zeros((300,))

        return embedding


def convert_words_into_embeddings(messages, content, embeddings_dict):
    def get_embeddings(ge_word_list, ge_content, ge_embeddings_dict):
        return list(map(lambda word: replace_word(ge_content, ge_embeddings_dict, word), ge_word_list))

    print('Se convirtieron las palabras al correspondiente embedding')

    return list(map(lambda word_list: get_embeddings(word_list, content, embeddings_dict), messages))


def obtain_tfidf_values(document_list):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(document_list)


def format_embedding_line(raw_line):
    elements = raw_line.split()
    del elements[0]

    elements = np.array(elements)
    elements = elements.astype(np.float)

    return elements


def generate_tuples_list(messages, tfidf_values, embeddings):
    def weight_embeddings(tfidf_list, embedding_list):
        new_embeddings = []
        for j in range(0, len(tfidf_list)):
            tfidf = tfidf_list[j]
            embedding = np.multiply(embedding_list[j], tfidf)
            new_embeddings.append(embedding)
        return new_embeddings

    tuples_list = []
    for i in range(0, len(messages)):
        """
            El primer elemento de la tupla es una lista de string (palabras)
            El segundo elemento es un np.array con los valores tfidf asociados
            El tercer elemento es una lista de np.array correspondiendo cada uno al embedding de cada palabra
        """
        tuples_list.append((messages[i], tfidf_values[i].data, embeddings[i]))

    tuples_list = list(map(lambda tupla: (tupla[0], tupla[1], weight_embeddings(tupla[1], tupla[2])), tuples_list))
    final_embeddings = list(map(lambda tupla: np.mean(tupla[2], axis=0), tuples_list))

    return np.array(final_embeddings)


# Genera un embedding para el string ingresado
def generate_embedding_from_sentence(sentence, content, embeddings_dict):
    word_list = sentence.split()
    embeddings = convert_words_into_embeddings([word_list], content, embeddings_dict)[0]
    tfidf = obtain_tfidf_values([sentence]).data

    weighted_embeddings = []
    for i in range(0, len(tfidf)):
        weighted_embeddings.append(np.multiply(embeddings[i], tfidf[i]))

    return [np.mean(np.array(weighted_embeddings), axis=0)]
