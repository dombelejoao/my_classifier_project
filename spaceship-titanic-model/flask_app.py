from flask import Flask, request, jsonify
from model import My_Classifier_Model

app = Flask(__name__)
model = My_Classifier_Model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Prediction logic
    return jsonify({"predictions": results})

if __name__ == '__main__':
    app.run(port=5000)
