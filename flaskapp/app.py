# from logging import log
from flask import Flask
from flask_ngrok import run_with_ngrok
from flask import session, url_for
# from flask_uploads import UploadSet
# from werkzeug.utils import secure_filename
# from contextlib import contextmanager
import json
import requests
# import click
# import sys, os
# import shutil
import json
from flask_cors import CORS
# import re
from flask import request
from flask import Flask, flash, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
# import smileScore # thay module ở đây bằng tên module của mọi người
# import os
# from tensorflow.keras.preprocessing.image import save_img
app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

UPLOAD_FOLDER = 'static/uploads/'
cors = CORS(app)
requested_site_url ='http://d1ae-34-134-131-99.ngrok.io'


app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['zip', 'rar'])



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():
    anchor_link = request.form['anchor_file']
    input_link = request.form['input_file']
    names_to_find = request.form['names_to_find']
    requested_site_url = request.form['backend_url']
    video_request = {'anchor_path' : anchor_link, 'input_path': input_link, 'people_names': names_to_find}
    video_request = json.dumps(video_request)
    res = requests.post(requested_site_url, json=video_request)
    res = res.text
    session['res'] = res
    return redirect(url_for('return_video_link', res=json.loads(res)))

    
@app.route('/handle_data/result', methods=['GET'])
def return_video_link():
    response = request.args['res']  # counterpart for url_for()
    response = session['res']       # counterpart for session
    print('1', flush=True)
    print(response, flush=True)
    return render_template("index.html", response=json.loads(response))


# @app.route('/display/<filename>')
# def display_image(filename):
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(debug=True)

# anchor link: https://drive.google.com/file/d/12q3_UwhKneoINYvgA-jcle6W2W8au9dE/view?usp=sharing
# input link:
# https://drive.google.com/file/d/1ARZQUff1AB6bCPEb5_SQjpTgRus900VR/view?usp=sharing
