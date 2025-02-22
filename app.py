from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the model and data
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    df = pd.read_csv('cleaned_car.csv')
except Exception as e:
    print(f"Error loading model or data: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        companies = sorted(df['company'].unique())
        car_models = sorted(df['name'].unique())
        year = sorted(df['year'].unique(), reverse=True)
        fuel_type = df['fuel_type'].unique()

        companies.insert(0, 'Select Company')
        return render_template('car.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)
    except Exception as e:
        # Handle errors and send appropriate message
        return f"An error occurred while loading the page: {e}"

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get form data
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        # Validate inputs
        if not company or not car_model or not year or not fuel_type or not driven:
            return "Missing required form data", 400

        # Convert driven to integer
        try:
            driven = int(driven)
        except ValueError:
            return "Kilometers driven must be a number", 400

        # Prepare data for prediction
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))

        # Predict the price
        prediction = model.predict(input_data)

        return jsonify({'predicted_price': str(np.round(prediction[0], 2))})

    except ValueError as ve:
        # Handle specific ValueError exceptions like incorrect input types
        return f"Value Error: {ve}"

    except Exception as e:
        # Handle all other exceptions
        return f"An error occurred during prediction: {e}"

if __name__ == '__main__':
    try:
        app.run(debug=True)  # Enable debug mode to provide more detailed error messages
    except Exception as e:
        print(f"Error starting the server: {e}")