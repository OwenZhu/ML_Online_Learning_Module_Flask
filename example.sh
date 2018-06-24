#!/bin/bash

# Starting API
python main.py &
sleep 2

# GET method train
curl -X GET http://localhost:5000/train && \
    echo -e "\n -> train OK"

# POST method partial train
curl -d '[
    {"problem_abstract": "Good morning", "Application_Status": "Active"},
    {"problem_abstract": "This is a simple test", "Application_Status": "Planned"},
    {"problem_abstract": "Tensorflow is good", "Application_Status": "Retired"},
]' -H "Content-Type: application/json" \
     -X POST http://localhost:5000/partial_train && \
    echo -e "\n -> predict OK"

# POST method predict
curl -d '[
    {"problem_abstract": "Good morning",
    {"problem_abstract": "This is a simple test",
    {"problem_abstract": "Tensorflow is good",
]' -H "Content-Type: application/json" \
     -X POST http://localhost:5000/predict && \
    echo -e "\n -> predict OK"

# GET method wipe
curl -X GET http://localhost:5000/wipe && \
    echo -e "\n -> wipe OK"

# kill runing API
for i in $(ps -elf | grep "python main.py" | grep -v grep | cut -d " " -f 4); do
    kill -9 $i
done