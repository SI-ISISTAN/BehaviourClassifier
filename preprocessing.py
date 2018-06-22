import numpy as np
import nltk
import corrector
from num2words import num2words

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

    with open(file_dir, mode='r', encoding='utf8') as file:
        content = file.read().splitlines()

    content = list(map(lambda line: split_line(line), content))
    classes = np.array(list(map(lambda tupla: int(tupla[0]), content)))
    messages = list(map(lambda tupla: tupla[1], content))

    print('Se cargó el archivo:', file_dir)

    return classes, messages


def generate_tfidf(messages):
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

    return tfidf, vectorizer


def obtain_embeddings_from_conversation(messages, embeddings_dict):
    palabras_no_encontradas = []

    stemmer = nltk.stem.SnowballStemmer('spanish')
    # corrector_ortografico = corrector.SpellCorrector('spanish.txt')
    corrector_ortografico = corrector.SpellCorrector('espanol.txt')

    def get_embedding(word):
        try:
            return embeddings_dict[word]
        except KeyError:
            try:
                return embeddings_dict[stemmer.stem(word)]
            except KeyError:
                try:
                    return embeddings_dict[corrector_ortografico.correct(word)]
                except KeyError:
                    try:
                        return embeddings_dict[corrector_ortografico.correct(stemmer.stem(word))]
                    except KeyError:
                        try:
                            return embeddings_dict[num2words(word, lang='es')]
                        except (KeyError, SyntaxError, NameError, TypeError):
                            nonlocal palabras_no_encontradas

                            palabras_no_encontradas.append(word)

                            return np.zeros((300,))

    def obtain_embeddings_from_sentence(sentence):
        return list(map(lambda word: get_embedding(word), sentence.split()))

    return list(map(lambda msg: obtain_embeddings_from_sentence(msg), messages)), palabras_no_encontradas


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

    return np.vstack(sentences2vec)


def get_embedding_for_predict(sentence, vectorizer, embeddings_dict):
    def get_tfidf_index(word, vocabulary):
        try:
            return vocabulary[word]
        except KeyError:
            return -1

    def get_tfidf_coef(word_index, matrix):
        try:
            return matrix.data[matrix.indices.tolist().index(word_index)]
        except ValueError:
            return np.float64(1.0)

    matrix = vectorizer.transform([sentence])
    sentence_embedding = obtain_embeddings_from_conversation([sentence], embeddings_dict)
    sentence_tfidf = list(map(lambda word: get_tfidf_index(word, vectorizer.vocabulary_), sentence.split()))
    sentence_tfidf = list(map(lambda index: get_tfidf_coef(index, matrix), sentence_tfidf))

    for i in range(0, len(sentence_tfidf)):
        sentence_embedding[0][i] = np.multiply(sentence_embedding[0][i], sentence_tfidf[i])

    final_embedding = np.mean(np.vstack(sentence_embedding[0]), axis=0).reshape(1, 300)

    return final_embedding


def get_reaccion(conducta):
    if conducta < 1 or conducta > 12:
        raise ValueError('Las conductas tienen un rango de entre 1 y 12. Valor ingresado: ->', conducta)

    mapeo_conducta_reaccion = {
        1: 'Positiva',
        2: 'Positiva',
        3: 'Positiva',
        4: 'Responde',
        5: 'Responde',
        6: 'Responde',
        7: 'Pregunta',
        8: 'Pregunta',
        9: 'Pregunta',
        10: 'Negativa',
        11: 'Negativa',
        12: 'Negativa',
    }

    return mapeo_conducta_reaccion[conducta]
