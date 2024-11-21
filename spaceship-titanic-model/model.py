import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Set up logging
logging.basicConfig(filename='./data/log_file.log', level=logging.INFO)

class My_Classifier_Model:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, dataset_filename):
        logging.info("Training started.")
        data = pd.read_csv(dataset_filename)
        # Preprocess data here...
        X = data.drop('Transported', axis=1)
        y = data['Transported']
        self.model.fit(X, y)
        joblib.dump(self.model, './model/model.pkl')
        logging.info("Model trained and saved.")

    def predict(self, dataset_filename):
        logging.info("Prediction started.")
        model = joblib.load('./model/model.pkl')
        data = pd.read_csv(dataset_filename)
        # Preprocess data here...
        predictions = model.predict(data)
        pd.DataFrame(predictions, columns=['Transported']).to_csv('./data/results.csv', index=False)
        logging.info("Predictions saved.")