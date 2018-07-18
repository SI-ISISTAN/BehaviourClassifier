from flask import Flask, jsonify, request
from model import ReactionNeuralNet

neuralnet = ReactionNeuralNet('result-studio/clasificador-reacciones_49/model.h5', 'embeddings.db')

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.run(host='localhost', port=8080)


@app.route('/clasificar', methods=['GET'])
def hello():
    mensaje_clasificar = request.args.get('mensaje', default='', type=str)

    if mensaje_clasificar == '':
        return jsonify(error='Se debe proveer un mensaje')

    return jsonify(
        mensaje=mensaje_clasificar,
        categoria=neuralnet.make_prediction(mensaje_clasificar)
    )
