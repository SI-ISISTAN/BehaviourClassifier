from keras.models import load_model

import preprocessing
import database_embeddings as embdb
import numpy as np


class ReactionNeuralNet:
    def __init__(self, weights_dir, database_dir):
        self.model = load_model(weights_dir)
        self.embeddings = embdb.EmbeddingDatabase(database_dir)
        self.categorias = {0: 'Negativa', 1: 'Positiva', 2: 'Pregunta', 3: 'Responde'}

    def make_prediction(self, raw_text):
        mensaje_clasificar = preprocessing.get_embedding_from_sentence(raw_text, self.embeddings)
        mensaje_clasificar = np.reshape(self.model.predict(mensaje_clasificar), (4,))
        mensaje_clasificar = preprocessing.one_hot(mensaje_clasificar)

        return self.categorias.get(int(np.argmax(mensaje_clasificar)))


# Ruta de los parámetros
# weights_dir = 'result-studio/clasificador-reacciones_49/model.h5'

# Cargo el .h5 con el modelo entrenado
# model = load_model(weights_dir)

# Obtengo la conexión con la base de los embeddings
# embeddings_db = demb.EmbeddingDatabase('embeddings.db')

# Obtengo el embedding asociado al mensaje provisto
# mensaje_clasificar = preprocessing.get_embedding_from_sentence('hola como andas', embeddings_db)

# prediction = np.reshape(model.predict(mensaje_clasificar), (4,))
# prediction = one_hot(softmax_pred)
