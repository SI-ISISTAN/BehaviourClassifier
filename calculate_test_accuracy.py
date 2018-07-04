import pandas as pd

# df = pd.read_csv('result-studio/prueba-cnn-hangouts-augmented_27/test_result.csv', sep=',')
# print('Test Accuracy: ', len(df.loc[df['categoria_reaccion'] == df['predictions']]) / len(df))

# df = pd.read_csv('result-studio/clasificador-conductas_38/result.csv', sep=',')
# print('Test Accuracy: ', len(df.loc[df['categoria_conducta'] == df['predictions']]) / len(df))

df = pd.read_csv('result-studio/clasificador-reacciones_49/result.csv', sep=',')
print('Test Accuracy: ', len(df.loc[df['categoria_reaccion'] == df['predictions']]) / len(df))
