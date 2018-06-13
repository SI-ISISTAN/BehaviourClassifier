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
                print('Van', id_embedding, 'de aprox. 1000000')

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

    print('Se cargó el archivo:', file_dir)

    return classes, messages


def obtain_tfidf_values(messages):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w*\b")
    sparse_matrix = vectorizer.fit_transform(messages)

    tfidf = []
    for i in range(0, sparse_matrix.shape[0]):
        word_list = messages[i].split()
        word_list = list(map(lambda word: vectorizer.vocabulary_[word], word_list))
        word_list = list(map(lambda word: sparse_matrix.indices.tolist().index(word), word_list))
        word_list = np.array(list(map(lambda word: sparse_matrix.data[word], word_list)))

        tfidf.append(word_list)

    print('Se terminó de armar la estructura de tfidf')

    return tfidf


def obtain_embeddings_from_conversation(messages, embeddings_dict):
    def get_embedding(word):
        try:
            return embeddings_dict[word]
        except KeyError:
            return np.zeros((300,))

    def obtain_embeddings_from_sentence(sentence):
        return list(map(lambda word: get_embedding(word), sentence.split()))

    return list(map(lambda msg: obtain_embeddings_from_sentence(msg), messages))


def obtain_sentences2vec(tfidf, embeddings):
    # Multiplico cada embedding por su correspondiente TFIDF
    for i in range(0, len(tfidf)):
        for j in range(0, len(tfidf[i])):
            embeddings[i][j] = np.multiply(embeddings[i][j], tfidf[i][j])

    print('Se arranca a armar el sentences2vec')

    # Para cada oración, armo el promedio de los vectores de cada palabra
    sentences2vec = []  # Sería una lista de numpy array con los vectores de cada ORACION
    for embedding in embeddings:
        sentences2vec.append(np.mean(np.array(embedding), axis=0))

    return sentences2vec
