import pandas as pd

df = pd.read_csv('result-studio/prueba-cnn-hangouts-augmented_12/result.csv', sep=',')

print('Test Accuracy: ', len(df.loc[df['categoria_reaccion'] == df['predictions']]) / len(df))
