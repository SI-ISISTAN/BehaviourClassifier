import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_embeddings(txt_dir):
    embedding_dict = {}
    id_embedding = 0

    with open(txt_dir, encoding='utf8') as file:
        file.readline()  # Para remover el encabezado
        for line in file:
            separator = line.index(' ')
            key = line[0:separator]
            values = np.fromstring(line[separator + 1:], dtype=float, sep=' ')
            embedding_dict[key] = values

            id_embedding += 1
            if id_embedding % 100000 == 0:
                print(id_embedding, 'de aprox. 1000000')

    print('Se generó el índice de embeddings')

    return embedding_dict


def load_data(file_dir):
    def split_line(line):
        aux = line.split('\t')
        return aux[0], aux[1]

    with open(file_dir, mode='r') as file:
        content = file.read().splitlines()

    content = list(map(lambda line: split_line(line), content))
    classes = np.array(list(map(lambda tupla: int(tupla[0]), content)))
    messages = list(map(lambda tupla: tupla[1], content))

    return classes, messages


def obtain_tfidf_values(document_list):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w*\b")
    sparse_matrix = vectorizer.fit_transform(document_list)

    tfidf = {}
    for i in range(0, sparse_matrix.shape[0]):
        tfidf[i] = sparse_matrix[i]

    return tfidf, vectorizer


def obtain_embeddings_from_conversation(messages, embeddings_dict):
    def get_embedding(word):
        try:
            return embeddings_dict[word]
        except KeyError:
            return np.zeros((300,))

    def obtain_embeddings_from_sentence(sentence):
        return list(map(lambda word: get_embedding(word), sentence.split()))

    return list(map(lambda msg: obtain_embeddings_from_sentence(msg), messages))


def obtain_sentences2vec(messages, tfidf_values, embeddings):
    sentences2vec = []  # Sería una lista de numpy array con los vectores de cada ORACION

    for i in range(0, len(embeddings)):
        tfidf = tfidf_values

