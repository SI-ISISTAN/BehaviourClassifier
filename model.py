import preprocessing
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gensim.models import Word2Vec


embeddings_dict = preprocessing.load_embeddings('datasets/SBW-vectors-300-min5.txt')
classes, messages = preprocessing.load_data('datasets/messages-hangouts.txt')
tfidf, vectorizer = preprocessing.generate_tfidf(messages)
embeddings = preprocessing.obtain_embeddings_from_conversation(messages, embeddings_dict)

# Armo el sentences2vec teniendo en cuenta TFIDF
sentences2vec = preprocessing.obtain_sentences2vec(tfidf, embeddings)

# Armo el sentences2vec sin tener el cuenta TFIDF
# sentences2vec = []
# for embedding in embeddings:
#     sentences2vec.append(np.mean(np.array(embedding), axis=0))
# sentences2vec = np.vstack(sentences2vec)

file = open('embeddings.csv', mode='w')
file.write('categoria,mensaje\n')
for i in range(0, sentences2vec.shape[0]):
    file.write(str(classes[i]) + ',' + ';'.join(list(map(lambda x: str(x), sentences2vec[i]))) + '\n')

file.close()


# classifier = SVC()
#
# X_train, X_test, y_train, y_test = train_test_split(sentences2vec, classes, test_size=0.3)
# print('Se empezó a entrenar el clasificador')
# classifier.fit(X_train, y_train)
#
# y_pred = classifier.predict(X_test)
# print('Accuracy: ', accuracy_score(y_test, y_pred))

# print(classifier.predict(preprocessing.get_embedding_for_predict('hola como andas', vectorizer, embeddings_dict)))

# TODO: empaquetar la traducción
# TODO: otra feature para agregar es pulir la oración a clasificar (por ejemplo volar las puntuaciones, acentos, etc.)
# TODO: bajar a disco el 'sentences2vec'
