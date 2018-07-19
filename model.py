from keras.models import load_model

import preprocessing
import database_embeddings as embdb
import numpy as np


class IPANeuralNet:
    def __init__(self, directories):
        self.clf_reacciones = load_model(directories.get('clf_reacciones'))
        self.clf_positiva = load_model(directories.get('clf_positiva'))
        self.clf_negativa = load_model(directories.get('clf_negativa'))
        self.clf_pregunta = load_model(directories.get('clf_pregunta'))
        self.clf_responde = load_model(directories.get('clf_responde'))

        self.embeddings = embdb.EmbeddingDatabase(directories.get('db_embeddings'))
        self.categorias = {
            0: {
                'reaccion': 'Negativa',
                'clasificador': self.clf_negativa,
                'conductas': {
                    0: 'C10 - Muestra Desacuerdo',
                    1: 'C11 - Muestra Tensión',
                    2: 'C12 - Muestra Antagonismo'
                }
            },
            1: {
                'reaccion': 'Positiva',
                'clasificador': self.clf_positiva,
                'conductas': {
                    0: 'C1 - Muestra Solidaridad',
                    1: 'C2 - Muestra Relajamiento',
                    2: 'C3 - Muestra Acuerdo'
                }
            },
            2: {
                'reaccion': 'Pregunta',
                'clasificador': self.clf_pregunta,
                'conductas': {
                    0: 'C7 - Pide Información',
                    1: 'C8 - Pide Opinión',
                    2: 'C9 - Pide Sugerencias'
                }
            },
            3: {
                'reaccion': 'Responde',
                'clasificador': self.clf_responde,
                'conductas': {
                    0: 'C4 - Da Sugerencias',
                    1: 'C5 - Da Opiniones',
                    2: 'C6 - Da Información'
                }
            }
        }

    def make_prediction(self, raw_text):
        resultado = {}

        # Genero el embedding asociado al texto plano
        embedding = preprocessing.get_embedding_from_sentence(raw_text, self.embeddings)

        # Obtengo la predicción de la REACCION
        softmax_prediction_reaccion = np.reshape(self.clf_reacciones.predict(embedding), (4,))
        onehot_prediction_reaccion = preprocessing.one_hot(softmax_prediction_reaccion)
        index_reaccion = int(np.argmax(onehot_prediction_reaccion))

        # Obtengo el clasificador asociado a la reacción predicha
        clasificador_conductas = self.categorias[index_reaccion]['clasificador']

        # Obtengo la predicción de la CONDUCTA
        softmax_prediction_conducta = np.reshape(clasificador_conductas.predict(embedding), (3,))
        onehot_prediction_conducta = preprocessing.one_hot(softmax_prediction_conducta)
        index_conducta = int(np.argmax(onehot_prediction_conducta))

        reaccion = self.categorias[index_reaccion]['reaccion']

        resultado['categoria'] = reaccion
        resultado['conducta'] = self.categorias[index_reaccion]['conductas'][index_conducta]

        return resultado

    def retrain(self, raw_text, conducta):

        """ Implementar lo siguiente:

            clf_reacciones.compile(optimizer='adadelta', loss='categorical_crossentropy')
            clf_conductas.compile(optimizer='adadelta', loss='categorical_crossentropy')

            USAR UN DICCIONARIO PARA:
                - Mapear cada conducta con una reacción
                - Mapear cada conducta con su onehot_vector
                - Mapear cada reacción con su onehot_vector

            clf_reacciones.fit(embedding, label_reaccion)
            clf_conductas.fit(embedding, label_conducta)

        """

        return 'Retrain sin implementar'
