from flask import Flask, jsonify, request
from model import IPANeuralNet

import configuration as conf

neuralnet = IPANeuralNet(conf.directories)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route('/')
def home():
    return jsonify(mensaje='que onda bigote')


@app.route('/clasificar', methods=['GET', 'POST'])
def clasificar():
    mensaje_clasificar = request.args.get('mensaje', default='', type=str)
    conducta = request.args.get('conducta', default=0, type=int)

    if mensaje_clasificar == '':
        return jsonify(error='Se debe proveer un mensaje')

    if request.method == 'GET':
        return jsonify(
            mensaje=mensaje_clasificar,
            resultado=neuralnet.make_prediction(mensaje_clasificar)
        )

    # Implementar para cuando haya que reentrenar
    if request.method == 'POST':
        if conducta == 0:
            return jsonify(error='Se debe proveer una conducta válida')

        return jsonify(
            mensaje=mensaje_clasificar,
            conducta_correcta=conducta,
            resultado=neuralnet.retrain(mensaje_clasificar, conducta)
        )


app.run(host=conf.server.get('address'), port=conf.server.get('port'))
