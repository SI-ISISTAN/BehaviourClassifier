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

    raw_text = file.decode('ISO-8859-1')

    cantidad_mensajes = int(request.args.get('cantidad_mensajes'))
    integrante_clasificar = request.args.get('integrante')

    csv = StringIO(raw_text)

    if cantidad_mensajes == -1:
        cantidad_mensajes = None

    mensajes_clasificar = pd.read_csv(csv, nrows=cantidad_mensajes, dtype=str)

    if integrante_clasificar is not None and integrante_clasificar is not '':
        mensajes_clasificar = pd.DataFrame(
            mensajes_clasificar[mensajes_clasificar['integrante'] == integrante_clasificar],
            dtype=str
        )

    headers_necesarios = ['chatId', 'timestamp', 'integrante', 'mensaje']

    headers = list(mensajes_clasificar.columns)
    headers = list(filter(lambda col: col in headers_necesarios, headers))

    if len(headers) != 4:
        return 'Se necesitan los siguientes encabezados: %s' % headers_necesarios

    def predict(msg):
        print('clasificó')
        return neuralnet.make_prediction(msg, nn_type)

    resultado = pd.DataFrame(
        list(map(lambda msg: predict(msg), mensajes_clasificar['mensaje'].values))
    )

    mensajes_clasificar['categoria'] = pd.Series(resultado['categoria'].values)
    mensajes_clasificar['conducta'] = pd.Series(resultado['conducta'].values)

    if integrante_clasificar is not None and integrante_clasificar is not '':
        mensajes_clasificar['categoria'] = resultado['categoria'].values
        mensajes_clasificar['conducta'] = resultado['conducta'].values

    output = StringIO()
    mensajes_clasificar.to_csv(output, index=False, na_rep='null')

    return Response(output.getvalue(), mimetype='text/csv')


if __name__ == '__main__':
    app.run()
