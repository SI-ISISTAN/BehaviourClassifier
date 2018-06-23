import preprocessing
import nltk
import gensim
import numpy as np

# classes, documents = preprocessing.load_data('datasets/messages-excel-v1-formatted.txt')
classes, documents = preprocessing.load_data('datasets/messages-hangouts.txt')

sentences = nltk.sent_tokenize(' '.join(documents), language='spanish')
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
word2vec_model = gensim.models.Word2Vec(sentences, size=1000, min_count=1)
print('Se armaron los embeddings')

tfidf, vectorizer = preprocessing.generate_tfidf(documents)
print('Se armaron los tfidf')

embeddings, cant_palabras_no_encontradas = preprocessing.obtain_embeddings_from_conversation(documents, word2vec_model)
print('Se mapearon los documentos a embeddings')

sentences2vec = preprocessing.obtain_sentences2vec(tfidf, embeddings)

# sentences2vec = []
# for embedding in embeddings:
#     sentences2vec.append(np.mean(np.array(embedding), axis=0))
# sentences2vec = np.vstack(sentences2vec)

file = open('embeddings-excel-v1-contfidf-hola.csv', mode='w')
# file = open('embeddings-hangouts-sintfidf.csv', mode='w')
file.write('categoria,mensaje\n')
for i in range(0, sentences2vec.shape[0]):
    file.write(str(classes[i]) + ',' + ';'.join(list(map(lambda x: str(x), sentences2vec[i]))) + '\n')

file.close()
