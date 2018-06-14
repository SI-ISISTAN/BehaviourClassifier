import preprocessing

from sklearn.naive_bayes import GaussianNB
import numpy as np

embeddings_dict = preprocessing.load_embeddings('datasets/SBW-vectors-300-min5.txt')
classes, messages = preprocessing.load_data('datasets/messages-hangouts.txt')
tfidf, vectorizer = preprocessing.generate_tfidf(messages)
embeddings = preprocessing.obtain_embeddings_from_conversation(messages, embeddings_dict)
sentences2vec = preprocessing.obtain_sentences2vec(tfidf, embeddings)

classifier = GaussianNB()
print('Se empezó a entrenar el clasificador')
classifier.fit(sentences2vec, classes)


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


sentence = 'decime como se hace esto'

print(classifier.predict(preprocessing.get_embedding_for_predict('hola como andas', vectorizer, embeddings_dict)))

# TODO: empaquetar la traducción
# TODO: fijarse de usar siempre el mismo vectorizer
# TODO: otra feature para agregar es pulir la oración a clasificar (por ejemplo volar las puntuaciones, acentos, etc.)

# messages_json, classes_json = preprocessing.load_data('chats-lotr.json')
# tfidf_values_json = preprocessing.obtain_tfidf_values(list(map(lambda msg_list: ' '.join(msg_list), messages_json)))
# embeddings_json = preprocessing.convert_words_into_embeddings(messages_json, content, embeddings_dict)
# sentences2vec_json = preprocessing.generate_tuples_list(messages_json, tfidf_values_json, embeddings_json)

# sentence_predict = generate_embedding_from_sentence('hola como andas', content, embeddings_dict)
#
# classifier = GaussianNB()
# classifier.fit(sentences2vec, classes)
# print(classifier.predict(sentence_predict))

# messages_arff, classes_arff = preprocessing.load_data_arff('chats-hangouts.arff')
# tfidf_values_arff = preprocessing.obtain_tfidf_values(list(map(lambda msg_list: ' '.join(msg_list), messages_arff)))
# embeddings_arff = preprocessing.convert_words_into_embeddings(messages_arff, content, embeddings_dict)
# sentences2vec_arff = preprocessing.generate_tuples_list(messages_arff, tfidf_values_arff, embeddings_arff)
