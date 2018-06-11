from preprocessing import load_data
from preprocessing import load_embeddings
from preprocessing import obtain_tfidf_values
from preprocessing import convert_words_into_embeddings
from preprocessing import generate_tuples_list
from preprocessing import generate_embedding_from_sentence

from sklearn.naive_bayes import GaussianNB

messages, classes = load_data('chats.json')
embeddings_dict, content = load_embeddings('SBW-vectors-300-min5.txt')
tfidf_values = obtain_tfidf_values(list(map(lambda msg_list: ' '.join(msg_list), messages)))
embeddings = convert_words_into_embeddings(messages, content, embeddings_dict)
sentences2vec = generate_tuples_list(messages, tfidf_values, embeddings)

sentence_predict = generate_embedding_from_sentence('hola como andas', content, embeddings_dict)

classifier = GaussianNB()
classifier.fit(sentences2vec, classes)
print(classifier.predict(sentence_predict))

# TODO: Cambiar los nombres de las funciones, acomodar
