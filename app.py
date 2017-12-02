from flask import Flask, render_template, request, redirect, url_for, make_response, session
import numpy as np
import pickle

app = Flask(__name__)

inputs = ['OFFTYPSB', 'NEWRACE', 'NEWEDUC', 'MONSEX', 'AGE', 'NEWCIT', 'DISTRICT', 'CRIMHIST']

with open('model.pkl', 'rb') as pickle_file:
	model = pickle.load(pickle_file)

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		return render_template('index.html')
	elif request.method == 'POST':
		arr = []
		for i in inputs:
			arr.append(int(request.form[i]))

		x = np.array(arr).reshape(1, -1)
		prediction = model.predict(x)
		return str(round(int(prediction[0]) / 12, 1)) + ' years'

if __name__ == '__main__':
	app.run()