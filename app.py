from flask import Flask, jsonify, request, Response
from model import IPANeuralNet
from io import StringIO

import configuration as conf

import pandas as pd

neuralnet = IPANeuralNet(conf.directories)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route('/')
def home():
    return jsonify(mensaje='server funcionando')


@app.route('/clasificar', methods=['GET'])
def clasificar():
    mensaje_clasificar = request.args.get('mensaje', default='', type=str)

    if mensaje_clasificar == '':
        return jsonify(error='Se debe proveer un mensaje')

    return jsonify(
        mensaje=mensaje_clasificar,
        resultado=neuralnet.make_prediction(mensaje_clasificar)
    )


@app.route('/reclasificar', methods=['POST'])
def reclasificar():
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


@app.route('/clasificarlote', methods=['POST'])
def clasificar_lote():
    csv = StringIO(request.get_data().decode('utf-8'))

    df = pd.read_csv(csv)
    print(df)

    return jsonify(
        mensaje='CSV Recibido'
    )


app.run(
    host=conf.server.get('address'),
    port=conf.server.get('port')
)

# app.run(
#     host='localhost',
# )

