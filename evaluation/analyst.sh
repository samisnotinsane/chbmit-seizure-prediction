#!/bin/bash

# Generate training and test dataset
# -------------------------
# python data_gen_train_test.py
# -------------------------

# chb01 training with ARMA
# -------------------------
# python analyst.py teach \
# --patient='chb01' --method='ARMA' --learning_algorithm='Linear SVM' --data='./data' \
# --learnersaveto='./models/chb01/AR'

# python analyst.py teach \
# --patient='chb01' --method='ARMA' --learning_algorithm='RBF SVM' --data='./data' \
# --learnersaveto='./models/chb01/AR'

# python analyst.py teach \
# --patient='chb01' --method='ARMA' --learning_algorithm='Logistic Regression' --data='./data' \
# --learnersaveto='./models/chb01/AR' --plot_figures
# -------------------------

# chb01 testing with ARMA
# -------------------------
python analyst.py think \
--patient='chb01' --method='ARMA' --learner='chb01_ARMA_Linear_SVM_v2' --train \
--data='./data' --models='./models' --saveto='./figures/chb01/AR' \
--saveformat='.pdf'

# chb06 training with ARMA
# -------------------------
# python analyst.py teach \
# --patient='chb06' --method='ARMA' --learning_algorithm='Linear SVM' --data='./data' \
# --learnersaveto='./models/chb06/AR'

# python analyst.py teach \
# --patient='chb06' --method='ARMA' --learning_algorithm='RBF SVM' --data='./data' \
# --learnersaveto='./models/chb06/AR'

# python analyst.py teach \
# --patient='chb06' --method='ARMA' --learning_algorithm='Logistic Regression' --data='./data' \
# --learnersaveto='./models/chb06/AR' --plot_figures
# -------------------------

# chb10 training with ARMA
# -------------------------
# python analyst.py teach \
# --patient='chb10' --method='ARMA' --learning_algorithm='Linear SVM' --data='./data' \
# --learnersaveto='./models/chb10/AR'

# python analyst.py teach \
# --patient='chb10' --method='ARMA' --learning_algorithm='RBF SVM' --data='./data' \
# --learnersaveto='./models/chb10/AR'

# python analyst.py teach \
# --patient='chb10' --method='ARMA' --learning_algorithm='Logistic Regression' --data='./data' \
# --learnersaveto='./models/chb10/AR' --plot_figures
# -------------------------

# chb01 prediction with ARMA
# -------------------------
# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='chb01_ARMA_Linear_SVM_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='chb01_ARMA_RBF_SVM_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='chb01_ARMA_Logistic_Regression_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf'
# -------------------------

# chb06 prediction with ARMA
# -------------------------
# python analyst.py think \
# --patient='chb06' --method='ARMA' --learner='chb06_ARMA_Linear_SVM_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb06/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb06' --method='ARMA' --learner='chb06_ARMA_RBF_SVM_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb06/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb06' --method='ARMA' --learner='chb06_ARMA_Logistic_Regression_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb06/AR' \
# --saveformat='.pdf'
# -------------------------

# chb10 prediction with ARMA
# -------------------------
# python analyst.py think \
# --patient='chb10' --method='ARMA' --learner='chb10_ARMA_Linear_SVM_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb10/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb10' --method='ARMA' --learner='chb10_ARMA_RBF_SVM_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb10/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb10' --method='ARMA' --learner='chb10_ARMA_Logistic_Regression_v2' \
# --data='./data' --models='./models' --saveto='./figures/chb10/AR' \
# --saveformat='.pdf'
# ------------------------- -------------------------  -------------------------

# Prediction in DEBUG mode
# -------------------------
# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='./models/chb01/AR/chb01_ARMA_Linear_SVM_v2.joblib' \
# --data='./data' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf' -- debug

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='./models/chb01/AR/chb01_ARMA_RBF_SVM_v2.joblib' \
# --data='./data' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf' --debug

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='./models/chb01/AR/chb01_ARMA_RBF_SVM_v2.joblib' \
# --data='./data' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf' --debug
