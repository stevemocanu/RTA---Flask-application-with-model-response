import pickle
import joblib
from flask import Flask
from flask import request
from flask import jsonify
from train import Perceptron

# Create a flask
app = Flask(__name__)

# Create an API end point
@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():

    # sepal length
    sepal_length = float(request.args.get('sl'))

    # petal length
    petal_length = float(request.args.get('pl'))

    # The features of the observation to predict
    features = [sepal_length,
                petal_length]

    # Load pickled model file
    model = joblib.load('model.pkl')

    # Predict the class using the model
    predicted_class = int(model.predict([features]))

    # Return a json object containing the features and prediction
    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    # Run the app at 0.0.0.0:3333
    app.run(port=3333,host='0.0.0.0')