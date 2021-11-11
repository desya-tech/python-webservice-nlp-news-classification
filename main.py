# from normalization import normalize_corpus
from flask import Flask, jsonify, request
# from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
from flasgger import Swagger
from flask_cors import CORS




app = Flask(__name__)
Swagger(app)
CORS(app)


@app.route('/input/task', methods=['POST'])
def predict():
    """
    Ini Adalah Endpoint Untuk Mengklasifikasi Lirik Lagu
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Lirik
          required:
            - text
          properties:
            text:
              type: string
              description: Please input with valid text.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()
    text = new_task['text']
    X_New = np.array([text])
   # X_New=normalize_corpus(X_New)


    pipe = joblib.load('naiveBayesClassifierBerita.pkl')

    resultBeritaPredict = pipe[0].predict(X_New)


    return jsonify({'message': format(resultBeritaPredict)})


if __name__ == '__main__':
    app.run(debug=True)
