from keras.models import load_model

import numpy as np
import preprocessing
import database_embeddings as demb


# Retorna el vector softmax discretizado
def one_hot(softmax_vector_output):
    one_hot_vector = np.zeros(softmax_vector_output.shape, dtype=np.float32)
    one_hot_vector[np.argmax(softmax_vector_output, axis=0)] = 1.0

    return one_hot_vector


# Ruta de los parámetros
weights_dir = 'result-studio/clasificador-reacciones_49/model.h5'

# Cargo el .h5 con el modelo entrenado
model = load_model(weights_dir)

# Obtengo la conexión con la base de los embeddings
embeddings_db = demb.EmbeddingDatabase('embeddings.db')

# Cargo el dataset
# classes, messages = preprocessing.load_data('datasets/raw/hangouts-augmented-v2.txt')

# Armo la matriz de tfidf correspondiente al dataset
# tfidf, vectorizer = preprocessing.generate_tfidf(messages)

mensaje_clasificar = 'hola como andas'
mensaje_clasificar = preprocessing.get_embedding_from_sentence(
    sentence=mensaje_clasificar, vectorizer=None, embeddings_db=embeddings_db
)


# prediction = np.reshape(model.predict(mensaje_clasificar), (4,))
# prediction = one_hot(softmax_pred)
