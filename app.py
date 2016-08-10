from flask import Flask, render_template, request, redirect, url_for,flash
from werkzeug.utils import secure_filename
import os
import alg
import numpy as np

app = Flask(__name__)
app.secret_key = "dsfj20jffdf"
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cl = {}
cln = {}
def load_lookup():
    f = open("coins.txt","r")
    line = 0
    for l in f:
        s = l.split(",")
        vec = np.array(list(map(float,s[1:])))
        cl[s[0]] = vec
    f = open("coinLookup.txt","r")
    for l in f:
        s = l.split("\t")
        cln[s[0]] = s[1]
load_lookup()

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/check', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        print("UPLOAD")
        print(request.files)
        if 'file' not in request.files:
            return ("Failed to find file in request")
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)
            vec = alg.get_file_vec(full_filename)
            cv = {}
            cc = {}
            for f in cl:
                dist = np.sum(cl[f]*vec)/(np.linalg.norm(cl[f])*np.linalg.norm(vec))
                if dist > 0.5:
                    ch = f[0:-8]
                    if not ch in cv or cv[ch] < dist:
                        cv[ch] = dist
                    if not ch in cc:
                        cc[ch] = 0
                    cc[ch] += 1
            items = sorted(cv, key=cv.get, reverse=True)[0:50]
            found = ",".join([cln[x] for x in items])
            return found
    return "Nothing found"
