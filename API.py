import pandas as pd
from sklearn.metrics import accuracy_score
from urllib import request, parse
import urllib.request
import json

df = pd.read_csv("C:\Geekbrains\s.csv")

X_test = df[['Title', 'Body']]
y_test = df['Y']


def get_prediction(title, body):
    res = {'Title': title, 'Body': body}

    myurl = "http://172.20.10.5:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(res)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))

    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']


predictions = X_test.iloc[200:210].apply(lambda x: get_prediction(x[0], x[1]), 1)


print(predictions)
print(accuracy_score(predictions, y_test[200:210]))

