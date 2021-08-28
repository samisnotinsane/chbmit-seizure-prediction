#!/bin/bash

python analyst.py teach \
--patient='chb01' --method='ARMA' --learning_algorithm='Linear SVM' --data='./data' \
--learnersaveto='./models/chb01/AR/chb01_AR_SVM_Linear_v2.joblib'

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='./models/chb01/AR/chb01_AR_SVM_Linear.joblib' \
# --data='./data' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf' --debug
