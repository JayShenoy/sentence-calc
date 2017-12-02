import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('datafull.csv')
variables = list(data)
variable_index = {variables[i]: i for i in range(len(variables))}
inputs = ['OFFTYPSB', 'NEWRACE', 'NEWEDUC', 'MONSEX', 'AGE', 'NEWCIT', 'DISTRICT', 'CRIMHIST']
output = 'TOTPRISN'

# Remove rows with NaN values
for i in inputs:
	data = data[pd.notnull(data[i])]
data = data[pd.notnull(data[output])]

arr_data = data.as_matrix().astype(int)
X = arr_data[:, [variable_index[i] for i in inputs]]
Y = arr_data[:, variable_index[output]]

idxs = np.random.randint(X.shape[0], size=50000)
X = X[idxs, :]
Y = Y[idxs]

new_index = {inputs[i]: i for i in range(len(inputs))}

model = RandomForestRegressor(n_estimators=30)
model.fit(X, Y)

with open('model2.pkl', 'wb') as pickle_file:
	pickle.dump(model, pickle_file)

print('Model stored in model.pkl')
print('R2 Value:', model.score(X, Y))
print('Feature Importances', model.feature_importances_)