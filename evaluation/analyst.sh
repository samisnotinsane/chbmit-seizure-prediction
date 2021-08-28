#!/bin/bash

python analyst.py think \
--patient='chb01' --method='ARMA' --learner='./models/chb01/AR/chb01_AR_SVM_Linear.joblib' \
--data='./data' --saveto='./figures/chb01/AR' \
--saveformat='.pdf' --debug
