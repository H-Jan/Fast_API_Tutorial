#Basic API that will return a simple message

#Important necessary modules
from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

#Declaring our FastAPI instance
app = FastAPI()

#Defining the Request Body and type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float 
    petal_width : float 

#Loading Iris Dataset
iris = load_iris()

#Getting Features and Targets
X = iris.data 
Y = iris.target 

#Fitting our model
clf = GaussianNB()
clf.fit(X,Y)

#Adding endpoint to our request body to predict the class of our attributes
@app.post('/predict')
def predict(data : request_body):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}


#Defining the path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to the Jungle!'}

#Defining the path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str):
    #Defining a function that takes only strings as input and outputs the following:
    return {'message': f'Welcome to the Jungle {name}'}
