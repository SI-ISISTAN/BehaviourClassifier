from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import load_data

messages, classes = load_data('chats.json')

vectorizer = TfidfVectorizer()
tfidf_values = vectorizer.fit_transform(list(map(lambda msg_list: ' '.join(msg_list), messages)))


