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


# def obtain_embeddings_from_conversation(messages, embeddings_dict):
#     palabras_no_encontradas = []
#
#     stemmer = nltk.stem.SnowballStemmer('spanish')
#     # corrector_ortografico = corrector.SpellCorrector('spanish.txt')
#     corrector_ortografico = corrector.SpellCorrector('lib/espanol.txt')
#
#     def get_embedding(word):
#         try:
#             return embeddings_dict[word]
#         except KeyError:
#             try:
#                 return embeddings_dict[stemmer.stem(word)]
#             except KeyError:
#                 try:
#                     return embeddings_dict[corrector_ortografico.correct(word)]
#                 except KeyError:
#                     try:
#                         return embeddings_dict[corrector_ortografico.correct(stemmer.stem(word))]
#                     except KeyError:
#                         try:
#                             return embeddings_dict[num2words(word, lang='es')]
#                         except (KeyError, SyntaxError, NameError, TypeError):
#                             nonlocal palabras_no_encontradas
#
#                             palabras_no_encontradas.append(word)
#
#                             return np.zeros((300,))
#
#     def obtain_embeddings_from_sentence(sentence):
#         return list(map(lambda word: get_embedding(word), sentence.split()))
#
#     return list(map(lambda msg: obtain_embeddings_from_sentence(msg), messages)), palabras_no_encontradas

def obtain_embeddings_from_conversation(messages, embeddings_db):
    palabras_no_encontradas = []

    stemmer = nltk.stem.SnowballStemmer('spanish')
    # corrector_ortografico = corrector.SpellCorrector('spanish.txt')
    corrector_ortografico = corrector.SpellCorrector('lib/espanol.txt')

    def get_embedding(word):
        try:
            return embeddings_db.get(word)
        except TypeError:
            try:
                return embeddings_db.get(stemmer.stem(word))
            except TypeError:
                try:
                    return embeddings_db.get(corrector_ortografico.correct(word))
                except TypeError:
                    try:
                        return embeddings_db.get(corrector_ortografico.correct(stemmer.stem(word)))
                    except TypeError:
                        try:
                            return embeddings_db.get(num2words(word, lang='es'))
                        except (TypeError, SyntaxError, NameError):
                            nonlocal palabras_no_encontradas

                            palabras_no_encontradas.append(word)

                            return np.zeros((300,))

    def obtain_embeddings_from_sentence(sentence):
        return list(map(lambda word: get_embedding(word), sentence.split()))

    return list(map(lambda msg: obtain_embeddings_from_sentence(msg), messages)), palabras_no_encontradas


def obtain_sentences2vec(tfidf_list, embeddings):
    if tfidf_list is None:
        return np.vstack(list(map(lambda x: np.mean(np.array(x), axis=0), embeddings)))

    # Multiplico cada embedding por su correspondiente TFIDF
    for i in range(0, len(tfidf_list)):
        for j in range(0, len(tfidf_list[i])):
            embeddings[i][j] = np.multiply(embeddings[i][j], tfidf_list[i][j])

    print('Se arranca a armar el sentences2vec')

    # Para cada oración, armo el promedio de los vectores de cada palabra
    sentences2vec = []  # Sería una lista de numpy array con los vectores de cada ORACION
    for embedding in embeddings:
        sentences2vec.append(np.mean(np.array(embedding), axis=0))

    return np.vstack(sentences2vec)


def get_embedding_from_sentence(sentence, embeddings_db, vectorizer=None):
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

    sentence_embedding = obtain_embeddings_from_conversation([sentence], embeddings_db)[0]

    if vectorizer is not None:
        matrix = vectorizer.transform([sentence])
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


def format_classifier_input(csv_dir):
    from keras.utils import np_utils

    with open(csv_dir, mode='r') as file:
        raw_content = file.readlines()

    del raw_content[0]
    raw_content = list(map(lambda line: line.split(','), raw_content))

    reaction_dict = {'Positiva': 0, 'Pregunta': 1, 'Responde': 2, 'Negativa': 3}

    reacciones = list(map(lambda line: np.float32(reaction_dict[line[0]]), raw_content))
    conductas = list(map(lambda line: np.float32(line[1]) - 1.0, raw_content))
    mensajes = list(map(lambda line: np.fromstring(line[2], dtype=np.float32, sep=';'), raw_content))

    reacciones = np_utils.to_categorical(reacciones, num_classes=4)
    conductas = np_utils.to_categorical(conductas, num_classes=12)
    mensajes = np.array(mensajes)

    reacciones_mensajes = list(map(lambda reaccion, mensaje: np.append(reaccion, mensaje), reacciones, mensajes))
    reacciones_mensajes = np.array(reacciones_mensajes)

    return reacciones, conductas, mensajes, reacciones_mensajes


def one_hot(softmax_vector_output):
    one_hot_vector = np.zeros(softmax_vector_output.shape, dtype=np.float32)
    one_hot_vector[np.argmax(softmax_vector_output, axis=0)] = 1.0

    return one_hot_vector

