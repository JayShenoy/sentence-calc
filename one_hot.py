import numpy as np

# Convert data to one-hot encoding scheme
codes = {}
for i in inputs:
	if i == 'AGE':
		continue

	col = X[:, new_index[i]]
	# codes contains indices of all possible values for each variable
	codes[i] = {}
	j = 0

	for entry in col:
		if not entry in codes[i]:
			codes[i][entry] = j
			j += 1

def one_hot_encode(variable, value):
	vec = [0 for _ in range(len(codes[variable]))]
	vec[codes[variable][value]] = 1
	return vec

def encode(arr):
	new_arr = [[] for _ in range(arr.shape[0])]

	for i in inputs:
		if i == 'AGE':
			continue

		col_idx = new_index[i]

		for j in range(arr.shape[0]):
			new_arr[j].extend(one_hot_encode(i, arr[j, col_idx]))

	# Append age to end of new_arr
	for j in range(arr.shape[0]):
		new_arr[j].append(arr[j, new_index['AGE']])

	return np.array(new_arr)