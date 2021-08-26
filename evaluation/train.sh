#!/bin/bash
echo 'Case: 1/3'
python3 chb01_AR_train.py
python3 chb01_PIB_train.py

echo 'Case: 2/3'
python3 chb06_AR_train.py
python3 chb06_PIB_train.py

echo 'Case: 3/3'
python3 chb10_AR_train.py
python3 chb10_PIB_train.py

echo 'Done'