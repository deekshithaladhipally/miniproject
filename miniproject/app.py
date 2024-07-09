import os
from flask import Flask, render_template, request
import pickle
import numpy as np

from flask import jsonify
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  
# Initialize Flask application instance named app
app = Flask(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

@app.route("/chat", methods=['GET'])
def chat_page():
    return render_template('chat.html')

@app.route("/send_message", methods=['POST'])
def send_message():
    user_message = request.json.get("message")
    response = get_gemini_response(user_message)
    bot_response = "".join(chunk.text for chunk in response)
    return jsonify({"response": bot_response})

# Function to load the model and make predictions
def predict(values):
    # Load the trained model
    model = pickle.load(open("models/breast_cancer.pkl", 'rb'))
    
    # Convert values to numpy array
    values = np.asarray(values)
    
    # Make prediction and return the result
    return model.predict(values.reshape(1, -1))[0]



# Route for home page
@app.route("/")
def home():
    return render_template('home.html')

# Route for cancer page (assuming it's a form page for input)
@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

# @app.route("/chat", methods=['GET', 'POST'])
# def chatPage():
#     return render_template('chat.html')

# Route for prediction page
@app.route("/predict", methods=['POST'])
def predictPage():
    if request.method == 'POST':
        # Get the input data as a dictionary
        to_predict_dict = request.form.to_dict()
        
        # Convert values to appropriate types (int or float)
        for key, value in to_predict_dict.items():
            try:
                to_predict_dict[key] = int(value)
            except ValueError:
                to_predict_dict[key] = float(value)
        
        # Extract values as a list and predict
        to_predict_list = list(to_predict_dict.values())
        
        # Call the predict function with values argument
        pred = predict(to_predict_list)
        
        # Render prediction result in predict.html
        return render_template('predict.html', pred=pred)

# Run the Flask web application
if __name__ == '__main__':
    app.run(debug=True)
