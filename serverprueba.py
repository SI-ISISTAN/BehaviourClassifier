from flask import Flask, jsonify, request
from io import StringIO

import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['POST'])
def recibir_lote():
    raw_text = request.get_data().decode('utf-8')
    csv = StringIO(raw_text)

    df = pd.read_csv(csv)
    df['clasificacion'] = pd.Series([6, 6, 6, 6, 6, 6])

    print(df)
    return jsonify(
        mensaje='CSV Recibido'
    )


if __name__ == '__main__':
    app.run()
