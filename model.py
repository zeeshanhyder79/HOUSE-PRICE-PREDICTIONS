import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
dataset = pd.DataFrame(pd.read_csv("./archive/Housing.csv"))
dataset[["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]] = dataset[["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]].replace({"yes":1,"no":0})
dataset[["furnishingstatus"]] = dataset[["furnishingstatus"]].replace({"furnished":0,"semi-furnished":1,"unfurnished":3})

furnishingstatus_col = dataset[["furnishingstatus"]]
encoder = OneHotEncoder()
encodedfurnishingstatus = encoder.fit_transform(furnishingstatus_col).toarray()
encoded_df = pd.DataFrame(encodedfurnishingstatus,columns=[f"furnishingstatus_{i}" for i in range(encodedfurnishingstatus.shape[1])])
dataset_encoded = pd.concat([dataset,encoded_df],axis=1)
dataset_encoded.drop("furnishingstatus", axis=1, inplace=True)
model = LinearRegression()
x = dataset_encoded.drop("price",axis=1)
y = dataset[["price"]]
model.fit(x,y)
pickle.dump(model,open("./Trained_model/housePriceModel",'wb'))