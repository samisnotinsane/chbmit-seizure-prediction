#!/bin/bash

python analyst.py teach \
--patient='chb01' --method='ARMA' --learning_algorithm='Linear SVM' --data='./data' \
--learnersaveto='./models/chb01/AR'

python analyst.py teach \
--patient='chb01' --method='ARMA' --learning_algorithm='RBF SVM' --data='./data' \
--learnersaveto='./models/chb01/AR'

python analyst.py teach \
--patient='chb01' --method='ARMA' --learning_algorithm='Logistic Regression' --data='./data' \
--learnersaveto='./models/chb01/AR'

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='./models/chb01/AR/chb01_AR_SVM_Linear.joblib' \
# --data='./data' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf' --debug
