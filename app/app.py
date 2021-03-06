from flask import Flask, request, render_template
from predict.predictor import predict
import numpy as np
from PIL import Image
import json
import re

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit')
def submit():
    return render_template('submit.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    print 'Grading Coin'
    image = Image.open(request.files['imagefile'])
    results = predict(image)
    # predictions = predict(imagefile)
    return render_template('result.html', data=results)

@app.route('/aboutme')
def aboutme():
    return render_template('aboutme.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
