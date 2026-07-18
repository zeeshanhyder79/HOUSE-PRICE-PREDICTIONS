from flask import Flask,render_template,request,redirect
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
app = Flask(__name__)

housePriceModel = pickle.load(open("./Trained_model/housePriceModel","rb"))

@app.route("/")
def home():
   return render_template("index.html")

@app.post("/calculatePrice")
def calculatePrice():
   if request.method == "POST":
      area = float(request.form['area'])
      bedrooms = int(request.form['bedrooms'])
      bathrooms = int(request.form['bathrooms'])
      stories = int(request.form['stories'])
      mainroad = request.form['mainroad']
      guestroom = request.form['guestroom']
      basement = request.form['basement']
      hotwater = request.form['hotwater']
      aircondition = request.form['aircondition']
      prefarea = request.form["prefarea"]
      furnishingstatus = request.form['furnishingstatus']
      parking = int(request.form['parking'])
   else:
      return redirect("/")
   
   data = {
       "area": [area],
       "bedrooms": [bedrooms],
       "bathrooms": [bathrooms],
       "stories": [stories],
       "mainroad": [mainroad],
       "guestroom": [guestroom],
       "basement": [basement],
       "hotwaterheating": [hotwater],
       "airconditioning": [aircondition],
       "furnishingstatus": [furnishingstatus],
       "parking": [parking],
       "prefarea": [prefarea]
   }
   df = pd.DataFrame(data)
   df[["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea"]] = df[["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea"]].replace({"yes": 1, "no": 0})
   df[["furnishingstatus"]] = df[["furnishingstatus"]].replace({"furnished": 0, "semi-furnished": 1, "unfurnished": 2})
   if df["furnishingstatus"].iloc[0] == 0:
      df = df.assign(furnishingstatus_0=1, furnishingstatus_1=0, furnishingstatus_2=0)
   elif df["furnishingstatus"].iloc[0] == 1:
      df = df.assign(furnishingstatus_0=0, furnishingstatus_1=1, furnishingstatus_2=0)
   else:
      df = df.assign(furnishingstatus_0=0, furnishingstatus_1=0, furnishingstatus_2=1)
   
   columns = ["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","hotwaterheating","airconditioning","parking","prefarea","furnishingstatus_0","furnishingstatus_1","furnishingstatus_2"]
   dataarr = df[columns]
   predication = housePriceModel.predict(dataarr)
   return render_template("index.html",data={'data': np.round(predication,2)})
   

if __name__== "__main__":
   app.run(debug = True)

