#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 15:20
# @Author  : Hanwei Zhu
# @File    : main.py

from flask import Flask
from flask import request, jsonify
import models
import config
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os

app = Flask(__name__)
app.config.from_object(config)

training_data_path = "data/training_data.csv"

clf = None
curr_version = 0


@app.route("/")
def index():
    return "Hello World!"


@app.route('/train', methods=['GET'])
def train():
    if request.method == 'GET':
        # load training set
        train_data = pd.read_csv('data/training_data.csv', index_col='id', encoding="ISO-8859-1")
        train_X = train_data['problem_abstract']
        train_y = train_data[['Active', 'Planned', 'Retired']]
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1, random_state=0)
        
        clf.fit(train_X, train_y)
        
        # TODO: validate the classifier, decide if save current version

        # save model after training
        global curr_version
        clf.save_model(str(curr_version))
        curr_version += 1

        # TODO: what to do when client waits the training result?
        pass


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if clf is None:
            return jsonify({'status': False, 'response': 'Train a model first'})

        formed_data = {"problem_abstract": "This is a test"}
        # json_ = request.json

        prediction = clf.predict(formed_data["problem_abstract"])
        return jsonify({"prediction": prediction})


@app.route('/rollback', methods=['POST'])
def rollback():
    global clf
    clf = models.RNNClassifier(app.config["MODEL_OPT"])
    global curr_version
    curr_version -= 3
    clf.start(curr_version, is_warm=True)


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree(app.config["model_dir"])
        os.makedirs(app.config["model_dir"])
        return jsonify({'status': True, 'response': 'Wipe successfully'})

    except Exception as e:
        print(e)
        return jsonify({'status': False, 'response': 'Could not remove and recreate the model directory'})


if __name__ == '__main__':
    clf = models.RNNClassifier(app.config["MODEL_OPTION"])
    app.run()
