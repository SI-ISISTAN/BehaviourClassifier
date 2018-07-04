"""
    Se tiene el siguiente (des)balance de clases:

    Archivo: hangouts-augmented-v5-contfidf.csv

    El CSV posee 4642 mensajes, los cuales originalmente eran la mitad, pero se usó
    data augmentation para mejorar la performance del clasificador. En concreto se
    hizo una traducción de cada mensaje al inglés, y de vuelta al español, resultando
    en ésta última traducción, los mensajes nuevos que se incorporaron al nuevo dataset.

    CONDUCTAS
    6 --> 1076
    4 --> 640
    3 --> 636
    5 --> 634
    1 --> 540
    7 --> 274
    8 --> 250
    2 --> 194
    11 --> 166
    10 --> 134
    9 --> 72
    12 --> 26

    REACCIONES
    Responde --> 2350
    Positiva --> 1370
    Pregunta --> 596
    Negativa --> 326

    Se harán las siguientes modificaciones para hacer balancear un poco el dataset:

    - Eliminar 500 instancias al azar de la clase 6
    - Eliminar 100 instancias al azar de la clase 4
    - Eliminar 200 instancias al azar de la clase 3
    - Eliminar 100 instancias al azar de la clase 5

    (Esto generará overfit, pero se puede compensar efectuando regularización)
    - Duplicar la cantidad de instancias de la reacción 'Pregunta'
    - Triplicar la cantidad de instancias de la reacción 'Negativa'

    Finalmente la distribución debería ser la siguiente:

    REACCIONES
    Responde --> 1650
    Positiva --> 1170
    Pregunta --> 1192
    Negativa --> 978

    CONDUCTAS
    6 --> 576 (antes 1076)
    4 --> 540 (antes 640)
    3 --> 436 (antes 636)
    5 --> 534 (antes 634)
    7 --> 548 (antes 274)
    8 --> 500 (antes 250)
    11 --> 498 (antes 166)
    10 --> 402 (antes 134)
    9 --> 144 (antes 72)
    12 --> 78 (antes 26)

    (Las clases 1 y 2 no se tocaron)
"""

import pandas as pd

# Cargo el CSV con los mensajes
df = pd.read_csv('datasets/preprocessed/hangouts-augmented5.csv', sep=',')

# Mezclo las instancias
df = df.sample(frac=1).reset_index(drop=True)

# Obtengo los mensajes de clase 6, los mezclo, y extraigo las primeras 500 instancias
df_6 = df.loc[df['categoria_conducta'] == 6]
df_6 = df_6.sample(frac=1).reset_index(drop=True)
df_6 = df_6.iloc[500:]

# Obtengo los mensajes de clase 4, los mezclo, y extraigo las primeras 100 instancias
df_4 = df.loc[df['categoria_conducta'] == 4]
df_4 = df_4.sample(frac=1).reset_index(drop=True)
df_4 = df_4.iloc[100:]

# Obtengo los mensajes de clase 3, los mezclo, y extraigo las primeras 200 instancias
df_3 = df.loc[df['categoria_conducta'] == 3]
df_3 = df_3.sample(frac=1).reset_index(drop=True)
df_3 = df_3.iloc[200:]

# Obtengo los mensajes de clase 5, los mezclo, y extraigo las primeras 100 instancias
df_5 = df.loc[df['categoria_conducta'] == 5]
df_5 = df_5.sample(frac=1).reset_index(drop=True)
df_5 = df_5.iloc[100:]

# Obtengo los mensajes de la clase 7, los duplico y los mezclo
df_7 = df.loc[df['categoria_conducta'] == 7]
df_7 = df_7.append(df_7)
df_7 = df_7.sample(frac=1).reset_index(drop=True)

# Obtengo los mensajes de la clase 8, los duplico y los mezclo
df_8 = df.loc[df['categoria_conducta'] == 8]
df_8 = df_8.append(df_8)
df_8 = df_8.sample(frac=1).reset_index(drop=True)

# Obtengo los mensajes de la clase 9, los duplico y los mezclo
df_9 = df.loc[df['categoria_conducta'] == 9]
df_9 = df_9.append(df_9)
df_9 = df_9.sample(frac=1).reset_index(drop=True)

# Obtengo los mensajes de la clase 10, los triplico y los mezclo
df_10 = df.loc[df['categoria_conducta'] == 10]
df_10 = df_10.append(df_10).append(df_10)
df_10 = df_10.sample(frac=1).reset_index(drop=True)

# Obtengo los mensajes de la clase 11, los triplico y los mezclo
df_11 = df.loc[df['categoria_conducta'] == 11]
df_11 = df_11.append(df_11).append(df_11)
df_11 = df_11.sample(frac=1).reset_index(drop=True)

# Obtengo los mensajes de la clase 12, los triplico y los mezclo
df_12 = df.loc[df['categoria_conducta'] == 12]
df_12 = df_12.append(df_12).append(df_12)
df_12 = df_12.sample(frac=1).reset_index(drop=True)

# Extraigo las clases restantes que no se tocan
df_1 = df.loc[df['categoria_conducta'] == 1]
df_2 = df.loc[df['categoria_conducta'] == 2]

# Uno todas las porciones y mezclo
df_nuevo = df_1.append(df_2).append(df_3).append(df_4)\
    .append(df_5).append(df_6).append(df_7).append(df_8)\
    .append(df_9).append(df_10).append(df_11).append(df_12)
df_nuevo = df_nuevo.sample(frac=1).reset_index(drop=True)

df_nuevo.to_csv('datasets/preprocessed/hangouts-augmented5.csv', sep=',', index=False)
