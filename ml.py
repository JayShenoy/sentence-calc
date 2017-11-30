import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv('data3k.csv', dtype=np.int32)
variables = list(data)
variable_index = {variables[i]: i for i in range(len(variables))}
inputs = ['OFFTYPSB', 'NEWRACE', 'NEWEDUC', 'MONSEX', 'AGE', 'NEWCIT', 'DISTRICT', 'CRIMHIST']
output = ['TOTPRISN']

arr_data = data.as_matrix()
X = arr_data[:, [variable_index[i] for i in inputs]]
Y = arr_data[:, [variable_index[i] for i in output]]

new_index = {inputs[i]: i for i in range(len(inputs))}

model = RandomForestRegressor(n_estimators=30)
model.fit(X, Y[:, 0])

with open('model.pkl', 'wb') as pickle_file:
	pickle.dump(model, pickle_file)