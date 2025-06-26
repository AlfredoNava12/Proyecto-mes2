from flask import Flask, request, render_template
import numpy as np
import joblib

# Crear la app Flask
app = Flask(__name__)

# Cargar el modelo de Red Neuronal (MLPRegressor)
model_mlp = joblib.load('model_mlp.pkl')

# Página principal con el formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para predecir con el modelo seleccionado
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    interest_rate = float(request.form['interest_rate'])
    inflation_gdp = float(request.form['inflation_gdp'])
    current_account = float(request.form['current_account'])
    gdp = float(request.form['gdp'])
    gni = float(request.form['gni'])

    # Crear el array de entrada para la predicción
    input_data = np.array([[interest_rate, inflation_gdp, current_account, gdp, gni]])

    # Hacer la predicción con el modelo MLP
    prediction = model_mlp.predict(input_data)[0]

    # Devolver el resultado de la predicción
    return render_template('index.html', prediction_text=f'Predicción de Inflación: {prediction:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)
