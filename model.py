from keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape, Conv1D, MaxPool1D, Flatten, Activation
from keras import optimizers
from keras.models import Model
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from datetime import datetime


def NeuralNet(input_shape):
    x_input = Input(input_shape)

    x = x_input

    x = Reshape(target_shape=(300, 1))(x)

    x = Conv1D(filters=32, kernel_size=2, padding='valid', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Conv1D(filters=16, kernel_size=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Conv1D(filters=8, kernel_size=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Conv1D(filters=4, kernel_size=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool1D(pool_size=4)(x)

    x = Flatten()(x)

    x = Dense(units=200, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = BatchNormalization()(x)

    x = Dense(units=100, activation='relu')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(units=50, activation='relu')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(units=20, activation='relu')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(units=10, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(units=4, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=x)

    return model


# with open('excel-v5-contfidf.csv', mode='r') as file:
#     raw_content = file.readlines()

# with open('hangouts-augmented-v5-contfidf.csv', mode='r') as file:
#     raw_content = file.readlines()

# with open('hangouts-augmented2.csv', mode='r') as file:
#     raw_content = file.readlines()

with open('datasets/preprocessed/hangouts-augmented3.csv', mode='r') as file:
    raw_content = file.readlines()

del raw_content[0]
raw_content = list(map(lambda line: line.split(','), raw_content))

reaction_dict = {'Positiva': 0, 'Pregunta': 1, 'Responde': 2, 'Negativa': 3}

reacciones = list(map(lambda line: np.float32(reaction_dict[line[0]]), raw_content))
conductas = list(map(lambda line: np.float32(line[1]) - 1.0, raw_content))
mensajes = list(map(lambda line: np.fromstring(line[2], dtype=np.float32, sep=';'), raw_content))

reacciones = np_utils.to_categorical(reacciones, num_classes=4)
conductas = np_utils.to_categorical(conductas, num_classes=12)
mensajes = np.array(mensajes)

# reacciones_mensajes = list(map(lambda reaccion, mensaje: np.append(reaccion, mensaje), reacciones, mensajes))
# reacciones_mensajes = np.array(reacciones_mensajes)

# X_train, X_test, y_train, y_test = train_test_split(mensajes, reacciones, test_size=0.2, shuffle=True)

# Acá arranca la ferretería de Deep Learning
# (Mensajes) ----> (Reacciones)
modelo_basico = NeuralNet((300,))

optimizer = optimizers.Adadelta(lr=1.0)

modelo_basico.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 20
history = modelo_basico.fit(x=mensajes, y=reacciones, validation_split=0.2, epochs=epochs, batch_size=32, verbose=2, shuffle=True)
# history = modelo_basico.fit(x=X_train, y=y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=2)

# Muestra sumario de la red neuronal
model_summary = modelo_basico.summary()

# Creo el directorio para almacenar el resultado
result_dir = './results/' + str(datetime.now().timestamp() * 1000000)
os.mkdir(result_dir)

# Muestro y guardo gráfico de accuracy para el training y validation set
plt.title('Accuracy History')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(result_dir + '/accuracy_graph.png', bbox_inches='tight')
plt.clf()

# Muestro y guardo gráfico de pérdida para el training y validation set
plt.title('Loss History')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(result_dir + '/loss_graph.png', bbox_inches='tight')

# Guardo el YAML con el modelo
model_yaml = modelo_basico.to_yaml()
with open(result_dir + '/model.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)

# Guardo el TXT con el modelo
with open(result_dir + '/model.txt', 'w') as txt_file:
    modelo_basico.summary(print_fn=lambda x: txt_file.write(x + '\n'))

# Creo el dataframe con los resultados
df = pd.DataFrame({
    'accuracy': history.history['acc'],
    'val_accuracy': history.history['val_acc'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

# Guardo el CSV con los resultados
df.to_csv(result_dir + '/results.csv', sep=',', encoding='utf8')

# # Genero métricas para los test
# metrics = modelo_basico.evaluate(x=X_test, y=y_test)

data_print = [
    result_dir,
    str(epochs),
    str(df['accuracy'].max()).replace('.', ','),
    str(df['loss'].max()).replace('.', ','),
    str(df['val_accuracy'].max()).replace('.', ','),
    str(df['val_loss'].max()).replace('.', ','),
    str(df['accuracy'].mean()).replace('.', ','),
    str(df['loss'].mean()).replace('.', ','),
    str(df['val_accuracy'].mean()).replace('.', ','),
    str(df['val_loss'].mean()).replace('.', ','),
    # str(metrics[1]).replace('.', ','),
    # str(metrics[0]).replace('.', ',')
]

print('\t'.join(data_print))

