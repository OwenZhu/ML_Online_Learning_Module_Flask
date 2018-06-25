#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 15:20
# @Author  : Hanwei Zhu
# @File    : main.py

from flask import Flask
from flask import request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os
import models
import config

app = Flask(__name__)
app.config.from_object(config)

training_data_path = "data/training_data.csv"
history_scores_path = 'data/history_scores.csv'

clf = None
curr_version = None
training_buffer = None


@app.route("/")
def index():
    return "Hello World!"


@app.route('/train', methods=['GET'])
def train():
    if request.method == 'GET':
        # load training set
        train_data = pd.read_csv(training_data_path, index_col='id', encoding="ISO-8859-1")
        train_X = train_data['problem_abstract'][:1024]
        train_y = train_data[['Active', 'Planned', 'Retired']][:1024]
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1, random_state=0)

        global clf
        clf.fit(train_X, train_y)

        # save model after training
        global curr_version
        curr_version += 1
        clf.save_model(str(curr_version))
        
        # validate the classifier
        clf.predict(val_X)
        score = clf.score(val_X, val_y)
        print("Accuracy is:", score)
        
        with open(history_scores_path, "a") as f:
            f.write(str(curr_version) + ',' + str(score) + '\n')
        
        # TODO: what to do when client waits the training result?
        return jsonify({"status": True, "response": "Finished training, accuracy is " + str(score)})


@app.route('/partial_train', methods=['POST'])
def partial_train():
    if request.method == 'POST':
        global curr_version
        if curr_version == -1:
            return jsonify({'status': False, 'response': 'Train a model first'})

        json_ = request.get_json()
        
        labels = ["Active", "Planned", "Retired"]
        
        text_list = []
        label_list = []
        for t in json_:
            if "problem_abstract" in t and "Application_Status" in t:
                text_list.append(t["problem_abstract"])
                l = [0, 0, 0]
                l[labels.index(t["Application_Status"])] = 1
                label_list.append(l)
        
        global clf
        clf.partial_fit(text_list, label_list)
        curr_version += 1
        
        clf.save_model(str(curr_version))
        
        return jsonify({"status": True, "response": "Finished partial training"})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if curr_version == -1:
            return jsonify({'status': False, 'response': 'Train a model first'})
        
        json_ = request.get_json()
        text_list = [t["problem_abstract"] for t in json_]
        prediction = clf.predict(text_list)
        
        labels = ["Active", "Planned", "Retired"]
        pred_str = [labels[p] for p in prediction]
        
        return jsonify({"status": True, "pred_list": pred_str})


@app.route('/rollback', methods=['GET'])
def rollback():
    if request.method == 'GET':
        global curr_version
        if curr_version < 3:
            return jsonify({"status": False, "response": "Cannot rollback"})

        global clf
        clf = models.RNNClassifier(app.config["MODEL_OPTION"])
        curr_version -= 3
        try:
            clf.start(curr_version, is_warm_start=True)
        except Exception as e:
            print(e)
            curr_version = -1
            clf.start()
            return jsonify({"status": False, "response": "Previous version not found, rollback to initial version"})

        return jsonify({"status": True, "response": "Rollback to version (" + str(curr_version) + ") successfully"})

    
@app.route('/get_history', methods=['GET'])
def get_history():
    scores = pd.read_csv(history_scores_path)
    ht = scores.to_json()
    return jsonify({"status": True, "response": ht})


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        # remove all model folders
        shutil.rmtree(app.config["MODEL_OPTION"]["model_dir"])
        os.makedirs(app.config["MODEL_OPTION"]["model_dir"])
        # remove tensorboard folder
        shutil.rmtree(app.config["MODEL_OPTION"]["tb_dir"])
        os.makedirs(app.config["MODEL_OPTION"]["tb_dir"])

        global curr_version, clf
        curr_version = -1
        clf = models.RNNClassifier(app.config["MODEL_OPTION"])
        clf.start()

        return jsonify({'status': True, 'response': 'Wipe successfully'})

    except Exception as e:
        print(e)
        return jsonify({'status': False, 'response': 'Could not remove and recreate the model directory'})


if __name__ == '__main__':
    
    # init version
    curr_version = -1
    
    # start model
    clf = models.RNNClassifier(app.config["MODEL_OPTION"])
    clf.start()
    print('model start')
    
    # TODO: create a partial training buffer
    # columns = ['Active', 'Planned', 'Retired']
    # training_buffer = pd.DataFrame(columns=columns)

    app.run(host='0.0.0.0', port=5000, debug=True)
