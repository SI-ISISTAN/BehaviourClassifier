from flask import Flask, request, Response
from io import StringIO
from model import IPANeuralNet

import pandas as pd
import configuration as conf

app = Flask(__name__)
neuralnet = IPANeuralNet(conf.directories)


@app.route('/', methods=['POST'])
def recibir_lote():
    csv = StringIO(request.get_data().decode('utf-8'))

    mensajes_clasificar = pd.read_csv(csv)

    headers_necesarios = ['id_sesion', 'timestamp', 'integrante', 'mensaje']

    headers = list(mensajes_clasificar.columns)
    headers = list(filter(lambda col: col in headers_necesarios, headers))

    if len(headers) != 4:
        return 'Se necesitan los siguientes encabezados: %s' % headers_necesarios

    resultado = pd.DataFrame(
        list(map(lambda msg: neuralnet.make_prediction(msg), mensajes_clasificar['mensaje'].values))
    )

    mensajes_clasificar['categoria'] = pd.Series(resultado['categoria'].values)
    mensajes_clasificar['conducta'] = pd.Series(resultado['conducta'].values)

    output = StringIO()
    mensajes_clasificar.to_csv(output, index=False)

    return Response(output.getvalue(), mimetype='text/csv')


if __name__ == '__main__':
    app.run()
