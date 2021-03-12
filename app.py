#from flask import Flask, render_template, request, redirect
import os  # new
from flask import Flask, send_file, jsonify, redirect
from flask import request, render_template
from flask_cors import CORS
import pandas as pd
import sklearn
import json, pickle
from sklearn.tree import DecisionTreeClassifier 
from joblib import load

app = Flask(__name__)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

my_dir = os.path.dirname( __file__)
model_file_path = os.path.join(my_dir, 'model.pickle')
scaler_file_path = os.path.join(my_dir, 'scaler.pickle')

@app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    """API request
    """
    if request.method == 'POST':  #this block is only entered when the form is submitted
        XGB = load('DesercionVPH.joblib')
        req_data = request.get_json () 
        if not req_data:
            return jsonify(error="request body cannot be empty"), 400
        LV_01 = 0
        LV_02 = 0
        IN_01 = 0
        EI_01 = 0
        AZ_01 = 0
        MA_01 = 0
        lugar_vacunacion = req_data['lugar_vacunacion']
        identidad_nacional = req_data['identidad_nacional']
        edad_inicio = req_data['edad_inicio']
        area_zona = req_data['area_zona']
        mes_acceso = req_data['mes_acceso']

        features = { 
            'lugar_vacunacion': lugar_vacunacion,
            'identidad_nacional': identidad_nacional,
            'edad_inicio': edad_inicio,
            'area_zona': area_zona,
            'mes_acceso': mes_acceso
        }

        if lugar_vacunacion == '1':
            LV_01 = 1
            LV_02 = 0
        if lugar_vacunacion== '2':
            LV_01 = 0
            LV_02 = 1
        if lugar_vacunacion == '3':
            LV_01 = 0
            LV_02 = 0
        if identidad_nacional == '1':
            IN_01 = 1
        if identidad_nacional == '2':
            IN_01 = 0
        EI_01 = int(edad_inicio)
        if area_zona == '1':
            AZ_01 = 1
        if area_zona == '2':
            AZ_01 = 0
        MA_01 = int(mes_acceso)

        Xnew = [LV_01,LV_02,IN_01,EI_01,AZ_01,MA_01]

        dataXnewValues = [["LV_01", "LV_02", "IN_01", "EI_01", "AZ_01", "MA_01"], Xnew]

        dataXnewColumns = dataXnewValues.pop(0)

        dataXnewDf = pd.DataFrame(dataXnewValues, columns=dataXnewColumns)

        Ynew = XGB.predict(dataXnewDf)

        if Ynew[0] == 0:
            Mensaje = 'Girl with high probability of no HPV vaccination dropout'
        else:
            Mensaje = 'Girl with high probability of HPV vaccination dropout'

        return jsonify( inputs=features,predictions=Mensaje)

    return '''User postman u otro cliente para ejecutar esta API REST'''

@app.route('/', methods=['GET','POST'])
def predict():
    """
    """
    if request.method == 'POST':  #this block is only entered when the form is submitted
        XGB = load('DesercionVPH.joblib')
        LV_01 = 0
        LV_02 = 0
        IN_01 = 0
        EI_01 = 0
        AZ_01 = 0
        MA_01 = 0
        lugar_vacunacion = request.form.get('lugar_vacunacion')
        identidad_nacional = request.form.get('identidad_nacional')
        edad_inicio = request.form.get('edad_inicio')
        area_zona = request.form.get('area_zona')
        mes_acceso = request.form.get('mes_acceso')
        features = { 
            'lugar_vacunacion': lugar_vacunacion,
            'identidad_nacional': identidad_nacional,
            'edad_inicio': edad_inicio,
            'area_zona': area_zona,
            'mes_acceso': mes_acceso
        }
        if lugar_vacunacion == '1':
            LV_01 = 1
            LV_02 = 0
        if lugar_vacunacion== '2':
            LV_01 = 0
            LV_02 = 1
        if lugar_vacunacion == '3':
            LV_01 = 0
            LV_02 = 0
        if identidad_nacional == '1':
            IN_01 = 1
        if identidad_nacional == '2':
            IN_01 = 0
        EI_01 = int(edad_inicio)
        if area_zona == '1':
            AZ_01 = 1
        if area_zona == '2':
            AZ_01 = 0
        MA_01 = int(mes_acceso)

        Xnew = [LV_01,LV_02,IN_01,EI_01,AZ_01,MA_01]

        dataXnewValues = [["LV_01", "LV_02", "IN_01", "EI_01", "AZ_01", "MA_01"], Xnew]

        dataXnewColumns = dataXnewValues.pop(0)

        dataXnewDf = pd.DataFrame(dataXnewValues, columns=dataXnewColumns)

        Ynew = XGB.predict(dataXnewDf)

        if Ynew[0] == 0:
            Mensaje = 'Girl with high probability of no HPV vaccination dropout'
        else:
            Mensaje = 'Girl with high probability of HPV vaccination dropout'
        return render_template("index.html", inputs=features, predictions=Mensaje, 
            lugar_vacunacion=lugar_vacunacion, identidad_nacional=identidad_nacional, 
            edad_inicio=edad_inicio, area_zona=area_zona, mes_acceso=mes_acceso)

    return render_template("index.html")

# puede eliminar desde esta línea en adelante
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify({'status': 'pong'})

@app.route('/<name>')
def hello_name(name):
    return "Hello {} {}!".format(name, sklearn.__version__)

if __name__ == "__main__":
    app.run()
    # app.run(debug=True)
