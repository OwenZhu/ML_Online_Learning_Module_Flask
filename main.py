#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/22 15:20
# @Author  : Hanwei Zhu
# @File    : main.py

from flask import Flask
from flask import request
import config

app = Flask(__name__)
app.config.from_object(config)

global_step = 0


@app.route("/")
def index():
    return "Hello World!"


@app.route('/train', methods=['GET'])
def train():
    pass


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_ = request.json
        pass


@app.route('/rollback', methods=['POST'])
def rollback():
    pass


if __name__ == '__main__':
    app.run()
