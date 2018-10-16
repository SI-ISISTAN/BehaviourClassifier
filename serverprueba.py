from flask import Flask, request, Response
from io import StringIO
from model import IPANeuralNet

import pandas as pd
import configuration as conf

app = Flask(__name__)
neuralnet = IPANeuralNet(conf.directories)


@app.route('/clasificar_csv/<nn_type>', methods=['POST'])
def clasificar_csv(nn_type):
    file = request.files['csv_file'].stream.read()

    # raw_text = file.decode('ISO-8859-1')
    raw_text = file.decode('utf-8')

    cantidad_mensajes = int(request.args.get('cantidad_mensajes'))
    integrantes_clasificar = request.args.get('integrante')

    # Control de valor de parámetros
    if cantidad_mensajes == -1:
        cantidad_mensajes = None

    if integrantes_clasificar == '' or integrantes_clasificar is None:
        integrantes_clasificar = []

    csv = StringIO(raw_text)

    # Leo el CSV entero con la cantidad de msjs especificada
    mensajes_clasificar = pd.read_csv(csv, nrows=cantidad_mensajes, dtype=str)

    # Si hay parámetros, separarlos
    if len(integrantes_clasificar) != 0:
        integrantes_clasificar = integrantes_clasificar.split(';')  # Se puede clasificar un grupo de personas
        mensajes_clasificar = pd.DataFrame(
            mensajes_clasificar[mensajes_clasificar['integrante'].isin(integrantes_clasificar)],
            dtype=str
        ).reset_index(drop=True)

    # Filtro los mensajes vacíos
    mensajes_clasificar = mensajes_clasificar[~mensajes_clasificar['mensaje'].str.isspace()]

    headers_necesarios = ['chatId', 'timestamp', 'integrante', 'mensaje']

    headers = list(mensajes_clasificar.columns)
    headers = list(filter(lambda col: col in headers_necesarios, headers))

    if len(headers) != 4:
        return 'Se necesitan los siguientes encabezados: %s' % headers_necesarios

    def predict(msg):
        return neuralnet.make_prediction(msg, nn_type)

    # resultado = pd.DataFrame(
    #     list(map(lambda msg: predict(msg), mensajes_clasificar['mensaje'].values))
    # )

    resultado = []

    contador = 0
    cantidad = len(mensajes_clasificar['mensaje'].values)

    for mensaje in mensajes_clasificar['mensaje'].values:
        print('Clasificando mensaje (%s de %s) USER: %s --> %s' %
              (contador, cantidad, mensajes_clasificar.iloc[contador]['integrante'], mensaje))
        resultado.append(predict(mensaje))
        contador += 1

    resultado = pd.DataFrame(resultado)

    mensajes_clasificar['categoria'] = pd.Series(resultado['categoria'].values)
    mensajes_clasificar['conducta'] = pd.Series(resultado['conducta'].values)

    output = StringIO()
    mensajes_clasificar.to_csv(output, index=False, na_rep='null')

    return Response(output.getvalue(), mimetype='text/csv')


if __name__ == '__main__':
    app.run()
