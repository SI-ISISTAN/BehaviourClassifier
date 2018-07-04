import preprocessing
import database_embeddings as demb

# embeddings_dict = preprocessing.load_embeddings('datasets/SBW-vectors-300-min5.txt')
# classes, messages = preprocessing.load_data('datasets/messages-excel-v1-formatted.txt')
# embeddings, palabras_no_encontradas = preprocessing.obtain_embeddings_from_conversation(messages, embeddings_dict)

# Obtengo la conexión con la base de los embeddings
embeddings_db = demb.EmbeddingDatabase('embeddings.db')

# Cargo el dataset
# classes, messages = preprocessing.load_data('datasets/raw/hangouts-augmented-v2.txt')
classes, messages = preprocessing.load_data('datasets/raw/messages-excel-v1-formatted.txt')

# Armo la matriz de tfidf correspondiente al dataset
# tfidf, vectorizer = preprocessing.generate_tfidf(messages)

# Genero un vector de embeddings para cada mensaje
embeddings, palabras_no_encontradas = preprocessing.obtain_embeddings_from_conversation(messages, embeddings_db)

# Armo el sentences2vec teniendo en cuenta TFIDF (Para no tenerlo en cuenta pasar None)
sentences2vec = preprocessing.obtain_sentences2vec(tfidf_list=None, embeddings=embeddings)

print('Se empezó a generar el CSV de salida')
file = open('datasets/preprocessed/hangouts-augmented5.csv', mode='w')
file.write('categoria_reaccion,categoria_conducta,mensaje\n')
for i in range(0, sentences2vec.shape[0]):
    file.write(str(preprocessing.get_reaccion(classes[i])) + ',' + str(classes[i]) + ',' +
               ';'.join(list(map(lambda x: str(x), sentences2vec[i]))) + '\n')

file.close()
