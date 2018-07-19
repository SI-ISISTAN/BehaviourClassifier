from flask import Flask, jsonify, request
from model import IPANeuralNet

import configuration as conf

neuralnet = IPANeuralNet(conf.directories.get('clf_reacciones'), conf.directories.get('db_embeddings'))

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.run(host=conf.server.get('address'), port=conf.server.get('port'))


@app.route('/clasificar', methods=['GET', 'POST'])
def clasificar():
    mensaje_clasificar = request.args.get('mensaje', default='', type=str)

    if mensaje_clasificar == '':
        return jsonify(error='Se debe proveer un mensaje')

    if request.method == 'GET':
        return jsonify(
            mensaje=mensaje_clasificar,
            categoria=neuralnet.make_prediction(mensaje_clasificar)
        )

    if request.method == 'POST':
        return 'Método POST sin implementar'
