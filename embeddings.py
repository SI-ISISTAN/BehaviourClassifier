
with open('SBW-vectors-300-min5.txt', encoding='utf8') as file:
    content = file.readlines()
del content[0]

print('se cargo el archivo')

embedding_dict = {}
id_embedding = 0
for word in list(map(lambda x: x.split()[0], content)):
    embedding_dict[word] = id_embedding
    id_embedding += 1

palabra = 'casa'

# Obtengo el embedding para la palabra ingresada (si no existe tira KeyError)
print(content[embedding_dict[palabra]])
