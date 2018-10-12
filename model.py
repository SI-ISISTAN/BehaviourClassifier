from keras.models import load_model

import preprocessing
import database_embeddings as embdb
import numpy as np


class IPANeuralNet:
    def __init__(self, directories):
        self.clf_reacciones = load_model(directories.get('clf_reacciones'))
        self.clf_reacciones.compile(optimizer='adadelta', loss='categorical_crossentropy')

        self.clf_reacciones_plain = load_model(directories.get('clf_reacciones_plain'))
        self.clf_reacciones_plain.compile(optimizer='adadelta', loss='categorical_crossentropy')

        self.clf_reacciones_cnn = load_model(directories.get('clf_reacciones_cnn'))
        self.clf_reacciones_cnn.compile(optimizer='adadelta', loss='categorical_crossentropy')

        self.clf_reacciones_rnn = load_model(directories.get('clf_reacciones_rnn'))
        self.clf_reacciones_rnn.compile(optimizer='adam', loss='categorical_crossentropy')

        self.clf_reacciones_crnn = load_model(directories.get('clf_reacciones_crnn'))
        self.clf_reacciones_crnn.compile(optimizer='adam', loss='categorical_crossentropy')

        self.clf_positiva = load_model(directories.get('clf_positiva'))
        self.clf_positiva.compile(optimizer='adadelta', loss='categorical_crossentropy')

        self.clf_negativa = load_model(directories.get('clf_negativa'))
        self.clf_negativa.compile(optimizer='adadelta', loss='categorical_crossentropy')

        self.clf_pregunta = load_model(directories.get('clf_pregunta'))
        self.clf_pregunta.compile(optimizer='adadelta', loss='categorical_crossentropy')

        self.clf_responde = load_model(directories.get('clf_responde'))
        self.clf_responde.compile(optimizer='adadelta', loss='categorical_crossentropy')

        self.clasificadores_reacciones = {
            'plain': self.clf_reacciones_plain,
            'cnn': self.clf_reacciones_cnn,
            'rnn': self.clf_reacciones_rnn,
            'crnn': self.clf_reacciones_crnn
        }

        # (conducta) -> (reaccion, clasificador_conducta, onehot_reaccion, onehot_conducta)
        self.mapeo_categoria_reaccion = {
            1: (1, self.clf_positiva, np.array([[0, 1, 0, 0]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32)),
            2: (1, self.clf_positiva, np.array([[0, 1, 0, 0]], dtype=np.float32), np.array([[0, 1, 0]], dtype=np.float32)),
            3: (1, self.clf_positiva, np.array([[0, 1, 0, 0]], dtype=np.float32), np.array([[0, 0, 1]], dtype=np.float32)),
            4: (3, self.clf_responde, np.array([[0, 0, 0, 1]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32)),
            5: (3, self.clf_responde, np.array([[0, 0, 0, 1]], dtype=np.float32), np.array([[0, 1, 0]], dtype=np.float32)),
            6: (3, self.clf_responde, np.array([[0, 0, 0, 1]], dtype=np.float32), np.array([[0, 0, 1]], dtype=np.float32)),
            7: (2, self.clf_pregunta, np.array([[0, 0, 1, 0]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32)),
            8: (2, self.clf_pregunta, np.array([[0, 0, 1, 0]], dtype=np.float32), np.array([[0, 1, 0]], dtype=np.float32)),
            9: (2, self.clf_pregunta, np.array([[0, 0, 1, 0]], dtype=np.float32), np.array([[0, 0, 1]], dtype=np.float32)),
            10: (0, self.clf_negativa, np.array([[1, 0, 0, 0]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32)),
            11: (0, self.clf_negativa, np.array([[1, 0, 0, 0]], dtype=np.float32), np.array([[0, 1, 0]], dtype=np.float32)),
            12: (0, self.clf_negativa, np.array([[1, 0, 0, 0]], dtype=np.float32), np.array([[0, 0, 1]], dtype=np.float32))
        }

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

    def make_prediction(self, raw_text, nn_type):
        resultado = {}

        # Genero el embedding asociado al texto plano
        embedding = preprocessing.get_embedding_from_sentence(raw_text, self.embeddings)

        # Obtengo el tipo de clasificador solicitado
        clasificador_reacciones = self.clasificadores_reacciones[nn_type]

        # Obtengo la predicción de la REACCION
        softmax_prediction_reaccion = np.reshape(clasificador_reacciones.predict(embedding), (4,))
        # softmax_prediction_reaccion = np.reshape(self.clf_reacciones.predict(embedding), (4,))
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

    def retrain(self, raw_text, conducta, epochs):

        reaccion, clf_conductas, onehot_reaccion, onehot_conducta = self.mapeo_categoria_reaccion[conducta]
        embedding = preprocessing.get_embedding_from_sentence(raw_text, self.embeddings)

        # Recompilo y reentreno el clasificador de la conducta asociada a la reacción
        # clf_conductas.compile(optimizer='adadelta', loss='categorical_crossentropy')
        clf_conductas.fit(embedding, onehot_conducta, epochs=epochs, verbose=0)

        # Recompilo y reentreno el clasificador de las reacciones
        # self.clf_reacciones.compile(optimizer='adadelta', loss='categorical_crossentropy')
        self.clf_reacciones.fit(embedding, onehot_reaccion, epochs=epochs, verbose=0)

        return 'Retrain efectuado'
