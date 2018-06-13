import preprocessing
import preprocessingNUEVO

from sklearn.naive_bayes import GaussianNB

embeddings_dict = preprocessingNUEVO.load_embeddings('datasets/SBW-vectors-300-min5.txt')
classes, messages = preprocessingNUEVO.load_data('datasets/messages-hangouts.txt')
tfidf = preprocessingNUEVO.obtain_tfidf_values(messages)
embeddings = preprocessingNUEVO.obtain_embeddings_from_conversation(messages, embeddings_dict)
sentences2vec = preprocessingNUEVO.obtain_sentences2vec(tfidf, embeddings)

classifier = GaussianNB()
print('Se empezó a entrenar el clasificador')
classifier.fit(sentences2vec, classes)

sentence_predict = preprocessingNUEVO.obtain_embeddings_from_conversation(['hola como andas'], embeddings_dict)
sentence_tfidf = preprocessingNUEVO.obtain_tfidf_values(sentence_predict)


# TODO: preparar la traducción de string a sentence2vec
# TODO: fijarse de usar siempre el mismo vectorizer

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
