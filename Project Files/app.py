from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('CCPP.pkl', 'rb'))

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/')
def home():
    return render_template('home.html')  # Serve home.html as the first page

@app.route('/index1')
def index1():
    return render_template('index1.html')  # Serve index1.html when requested

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from frontend

    # Extract input features
    AT = float(data['AT'])
    V = float(data['V'])
    AP = float(data['AP'])
    RH = float(data['RH'])

    # Make prediction
    input_data = np.array([[AT, V, AP, RH]])
    prediction = model.predict(input_data)

    return jsonify({'Predicted Power Output': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
