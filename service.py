from flask import Flask, json, request
from utils import predict_bonus, predict_neighbours
import pandas as pd
import pickle
import zipfile
import os

gbc_clf = None
nbrs_clf = None
service_params = None
client_db = None
merchant_db = None

companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

api = Flask(__name__)

@api.route('/predict_bonus', methods=['GET'])
def get_predict_bonus():
  args = request.args
  bonus = []
  if 'client_id' in args and 'store_id' in args and 'rating' in args:
    bonus, prob = predict_bonus(int(args['client_id']), int(args['store_id']), float(args['rating']) gbc_clf, client_db, merchant_db, service_params)
    return json.dumps([{"bonus": bonus, 'probability': prob}])
  else:
    return "Missing values: client_id, store_id, rating"

@api.route('/predict_neighbours', methods=['GET'])
def get_predict_neighbours():
    args = request.args
    neighbours = ''
    if 'client_id' in args:
      neighbours, similarity = predict_neighbours(int(args['client_id']), nbrs_clf, client_db, service_params)
    return json.dumps([{"neighbours": neighbours.tolist(), 'similarity': similarity}])
  
@api.route('/health', methods=['GET'])
def health():
    return "OK"

if __name__ == '__main__':
  with zipfile.ZipFile('data/gbc.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')
  with zipfile.ZipFile('data/nbrs.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')
  gbc_clf = pickle.load(open('data/gbc.clf', 'rb'))
  nbrs_clf = pickle.load(open('data/nbrs.clf', 'rb'))
  service_params = pickle.load(open('data/service.params', 'rb'))
  client_db = pd.read_csv('data/test_db.csv')
  merchant_db = pd.read_csv('data/merchants_db.csv')
  port = os.getenv('PORT')
  port = 5000 if not port else port
  try:
    api.run(host='0.0.0.0', port=port)
  except:
    print("failed to run on port 80, rerun on default")