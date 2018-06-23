
from keras.preprocessing.text import Tokenizer

# Se definen cinco documentos
documents = [
    'Hola cómo andas?',
    'Todo bien y vos?',
    'Me parece que todo tranquilo',
    'Por?',
    'Buena pregunta como la otra... veremos jaja']

# Se instancia el Tokenizer
tokenizer = Tokenizer()

# Se pasan los documentos por el Tokenizer
tokenizer.fit_on_texts(documents)

# Parecido a word_docs
print('Word Counts:', tokenizer.word_counts)

# Cantidad de documentos
print('Document Count:', tokenizer.document_count)

# A cada palabra le asigna un identificador entero
print('Word Index:', tokenizer.word_index)

# Cuenta la cantidad de cada palabra en el conjunto de documentos
print('Word Documents:', tokenizer.word_docs)

documentos_encodeados = tokenizer.texts_to_matrix(documents, mode='count')
print(documentos_encodeados)

# Traduce cada palabra a su índice (si una palabra no está en el tokenizer no lo traduce)
print(tokenizer.texts_to_sequences(['como andas pibe']))


