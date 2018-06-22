import preprocessing
import numpy as np


embeddings_dict = preprocessing.load_embeddings('datasets/SBW-vectors-300-min5.txt')
# classes, messages = preprocessing.load_data('datasets/messages-excel-v1-formatted.txt')
# classes, messages = preprocessing.load_data('datasets/messages-hangouts-augmented.txt')
classes, messages = preprocessing.load_data('datasets/hangouts-augmented2.csv')
tfidf, vectorizer = preprocessing.generate_tfidf(messages)
embeddings, palabras_no_encontradas = preprocessing.obtain_embeddings_from_conversation(messages, embeddings_dict)

# Armo el sentences2vec teniendo en cuenta TFIDF
sentences2vec = preprocessing.obtain_sentences2vec(tfidf, embeddings)

# Armo el sentences2vec sin tener el cuenta TFIDF
# sentences2vec = []
# for embedding in embeddings:
#     sentences2vec.append(np.mean(np.array(embedding), axis=0))
# sentences2vec = np.vstack(sentences2vec)

print('Se empezó a generar el CSV de salida')
file = open('hangouts-augmented2.csv', mode='w')
file.write('categoria_reaccion,categoria_conducta,mensaje\n')
for i in range(0, sentences2vec.shape[0]):
    file.write(str(preprocessing.get_reaccion(classes[i])) + ',' + str(classes[i]) + ',' +
               ';'.join(list(map(lambda x: str(x), sentences2vec[i]))) + '\n')

file.close()
