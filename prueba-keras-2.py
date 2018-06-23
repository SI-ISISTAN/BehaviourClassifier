import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM


def split_line(line):
    separator = line.index('\t')
    category = line[separator + 1:].replace('\n', '')
    message = line[0:separator]
    return category, message


def get_padding(word_list, max_length):
    remaining = max_length - len(word_list)
    return [0] * remaining


with open('datasets/messages-excel-v1-formatted.txt') as file:
    content = file.readlines()

content = list(map(lambda line: split_line(line), content))
documents = list(map(lambda line: line[0], content))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)

# X = tokenizer.texts_to_matrix(documents)
X = tokenizer.texts_to_sequences(documents)

for word_list in X:
    word_list.extend([0] * (180 - len(word_list)))

X = np.vstack(X)
y = list(map(lambda line: int(line[1]), content))


# Modelo
model = Sequential()
model.add(Embedding(input_dim=5709, output_dim=300, input_length=180))
model.add(LSTM(units=12))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(x=X, y=y, epochs=50, verbose=1, shuffle=True)
loss, accuracy = model.evaluate(X, y, verbose=1)
print('Accuracy: %f' % (accuracy*100))


# file = open('aburuba.csv', mode='w')
# file.write('categoria,mensaje\n')
# for i in range(0, X.shape[0]):
#     file.write(str(y[i]) + ',' + ';'.join(list(map(lambda example: str(example), X[i]))) + '\n')
#
# file.close()
