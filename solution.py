import torch 
from torch import nn
from torch import functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from matplotlib import pyplot as plt

try:
  import catboost as cb
except:
  !pip install catboost
  import catboost as cb
  
  SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
#random.seed(SEED)

!wget https://www.dropbox.com/s/1kz4rj67xna7ta7/test.csv
!wget https://www.dropbox.com/s/b99noaytid2fcnr/train.csv
!wget https://www.dropbox.com/s/03oq740813ndhj9/sample_submit.csv

test_raw = pd.read_csv('/content/test.csv') 
train_raw = pd.read_csv('/content/train.csv')
target = train_raw.y

print("\n".join([i for i in train_raw.columns]))

def transform_data(data):
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  data = data.fillna(0).reset_index()
  data = data.select_dtypes(include=numerics)

  fun = data["Education Index"]*10 + data["Unemployment, total (% of labour force)"]*4 - data["Education Index"]*15
  funname = "fun"
  data = pd.merge(data , pd.DataFrame(fun, columns=[funname]).reset_index(), on="index")
  #data.apply(np.exp)
  return data
  train = transform_data(train_raw)
test = transform_data(test_raw)
common_features=list(set(train.columns).intersection(set(test.columns)))
common_features.remove("index") # это список, содержащий колонки, которые есть как в трейн, так и в тест дате

X_train, X_val, y_train, y_val = train_test_split(train[common_features], target, test_size=0.3, random_state=SEED) 

lr = 1e-3

model = cb.CatBoostRegressor(
    iterations=3000,
    eval_metric='RMSE',
    task_type='CPU',
    learning_rate=0.03
  )
model.fit(
      X_train, y_train,
      eval_set=(X_val,y_val),
      verbose=300
  )
  
  def show_model_fi(model):
  feature_im = model.feature_importances_
  importances = {}
  for value, feature in zip(feature_im, common_features):
    importances[feature] = value
  importances = sorted(importances.items(), key = lambda x: x[1])
  plt.style.use('fivethirtyeight')
  plt.figure(figsize=(13, 30))
  plt.barh([x[0] for x in importances], [x[1] for x in importances])
  
  show_model_fi(model)
  
from sklearn.metrics import mean_squared_error
from math import sqrt

rms_stack = sqrt(mean_squared_error(y_val, stack_preds))
preds_one = model.predict(X_val)

rms_one = sqrt(mean_squared_error(y_val, preds_one))

print(f'Stack: {rms_stack}\nOne model: {rms_one}')

ex = pd.read_csv('/content/sample_submit.csv')

ex.head()

import os

submission_name = input('Submission name:\n')
submission_path = 'submissions/{}.csv'.format(submission_name)

if not os.path.exists('submissions'):
    os.makedirs('submissions')

print(submission_path)
sub.to_csv(submission_path, index=True)
