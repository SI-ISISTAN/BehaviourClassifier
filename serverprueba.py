from flask import Flask, request, Response
from io import StringIO
from model import IPANeuralNet

import pandas as pd
import configuration as conf

app = Flask(__name__)
neuralnet = IPANeuralNet(conf.directories)


@app.route('/clasificar_csv', methods=['POST'])
def clasificar_csv():
    csv = StringIO(request.files['csv_file'].stream.read().decode('utf-8'))
    cantidad_mensajes = int(request.args.get('cantidad_mensajes'))

    mensajes_clasificar = pd.read_csv(csv, nrows=cantidad_mensajes)

    headers_necesarios = ['chatId', 'timestamp', 'integrante', 'mensaje']

    headers = list(mensajes_clasificar.columns)
    headers = list(filter(lambda col: col in headers_necesarios, headers))

    if len(headers) != 4:
        return 'Se necesitan los siguientes encabezados: %s' % headers_necesarios

    def predict(msg):
        print('clasificó')
        return neuralnet.make_prediction(msg)

    resultado = pd.DataFrame(
        # list(map(lambda msg: neuralnet.make_prediction(msg), mensajes_clasificar['mensaje'].values))
        list(map(lambda msg: predict(msg), mensajes_clasificar['mensaje'].values))
    )

    mensajes_clasificar['categoria'] = pd.Series(resultado['categoria'].values)
    mensajes_clasificar['conducta'] = pd.Series(resultado['conducta'].values)

    output = StringIO()
    mensajes_clasificar.to_csv(output, index=False)

    return Response(output.getvalue(), mimetype='text/csv')


# @app.route('/clasificar_arff', methods=['POST'])
# def clasificar_arff():
#     arff = StringIO(request.get_data().decode('utf-8'))


if __name__ == '__main__':
    app.run()
