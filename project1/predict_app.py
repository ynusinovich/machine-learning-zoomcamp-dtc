from flask import Flask
from flask import request
from flask import jsonify

from predict import predict

app = Flask('molecule')


@app.route('/predict_app', methods=['POST'])
def predict_app():
    molecule = request.get_json()

    result = predict(molecule)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)