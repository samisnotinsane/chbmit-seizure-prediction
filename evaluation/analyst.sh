#!/bin/bash

# Generate training and test dataset
# -------------------------
# python data_gen_train_test.py
# -------------------------

echo 'Invoking...'

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

# chb01 testing with ARMA
# -------------------------
# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='chb01_ARMA_Linear_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='chb01_ARMA_RBF_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='chb01_ARMA_Logistic_Regression_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf'
# -------------------------

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

# chb06 testing with ARMA
# -------------------------
# python analyst.py think \
# --patient='chb06' --method='ARMA' --learner='chb06_ARMA_Linear_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb06/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb06' --method='ARMA' --learner='chb06_ARMA_RBF_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb06/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb06' --method='ARMA' --learner='chb06_ARMA_Logistic_Regression_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb06/AR' \
# --saveformat='.pdf'
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

# chb10 testing with ARMA
# -------------------------
# python analyst.py think \
# --patient='chb10' --method='ARMA' --learner='chb10_ARMA_Linear_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb10/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb10' --method='ARMA' --learner='chb10_ARMA_RBF_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb10/AR' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb10' --method='ARMA' --learner='chb10_ARMA_Logistic_Regression_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb10/AR' \
# --saveformat='.pdf'
# -------------------------

# chb01 prediction with ARMA (TEST set)
# -------------------------
python analyst.py think \
--patient='chb01' --method='ARMA' --learner='chb01_ARMA_Linear_SVM_v2' \
--data='./data' --models='./models' --saveto='./figures/chb01/AR' \
--saveformat='.pdf'

python analyst.py think \
--patient='chb01' --method='ARMA' --learner='chb01_ARMA_RBF_SVM_v2' \
--data='./data' --models='./models' --saveto='./figures/chb01/AR' \
--saveformat='.pdf'

python analyst.py think \
--patient='chb01' --method='ARMA' --learner='chb01_ARMA_Logistic_Regression_v2' \
--data='./data' --models='./models' --saveto='./figures/chb01/AR' \
--saveformat='.pdf'
# -------------------------

# chb06 prediction with ARMA (TEST set)
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
# -------------------------

# Prediction in DEBUG mode with ARMA
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

# ------------------------- -------------------------  -------------------------

# chb01 training with Spectral Power in Band
# -------------------------------------------------- 
# python analyst.py teach \
# --patient='chb01' --method='Spectral' --learning_algorithm='Linear SVM' --data='./data' \
# --learnersaveto='./models/chb01/Spectral'

# python analyst.py teach \
# --patient='chb01' --method='Spectral' --learning_algorithm='RBF SVM' --data='./data' \
# --learnersaveto='./models/chb01/Spectral'

# python analyst.py teach \
# --patient='chb01' --method='Spectral' --learning_algorithm='Logistic Regression' --data='./data' \
# --learnersaveto='./models/chb01/Spectral' --plot_figures

# chb01 testing with Spectral Power in Band
# -------------------------
# python analyst.py think \
# --patient='chb01' --method='Spectral' --learner='chb01_Spectral_Linear_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb01/Spectral' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb01' --method='Spectral' --learner='chb01_Spectral_RBF_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb01/Spectral' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb01' --method='Spectral' --learner='chb01_Spectral_Logistic_Regression_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb01/Spectral' \
# --saveformat='.pdf'
# -------------------------------------------------- 

# chb06 training with Spectral Power in Band
# -------------------------------------------------- 
# python analyst.py teach \
# --patient='chb06' --method='Spectral' --learning_algorithm='Linear SVM' --data='./data' \
# --learnersaveto='./models/chb06/Spectral' --plot_figures

# python analyst.py teach \
# --patient='chb06' --method='Spectral' --learning_algorithm='RBF SVM' --data='./data' \
# --learnersaveto='./models/chb06/Spectral'

# python analyst.py teach \
# --patient='chb06' --method='Spectral' --learning_algorithm='Logistic Regression' --data='./data' \
# --learnersaveto='./models/chb06/Spectral'

# chb06 testing with Spectral Power in Band
# -------------------------
# python analyst.py think \
# --patient='chb06' --method='Spectral' --learner='chb06_Spectral_Linear_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb06/Spectral' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb06' --method='Spectral' --learner='chb06_Spectral_RBF_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb06/Spectral' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb06' --method='Spectral' --learner='chb06_Spectral_Logistic_Regression_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb06/Spectral' \
# --saveformat='.pdf'
# -------------------------------------------------- 

# chb10 training with Spectral Power in Band
# -------------------------------------------------- 
# python analyst.py teach \
# --patient='chb10' --method='Spectral' --learning_algorithm='Linear SVM' --data='./data' \
# --learnersaveto='./models/chb10/Spectral' --plot_figures

# python analyst.py teach \
# --patient='chb10' --method='Spectral' --learning_algorithm='RBF SVM' --data='./data' \
# --learnersaveto='./models/chb10/Spectral'

# python analyst.py teach \
# --patient='chb10' --method='Spectral' --learning_algorithm='Logistic Regression' --data='./data' \
# --learnersaveto='./models/chb10/Spectral'

# chb06 testing with Spectral Power in Band
# -------------------------
# python analyst.py think \
# --patient='chb10' --method='Spectral' --learner='chb10_Spectral_Linear_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb10/Spectral' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb10' --method='Spectral' --learner='chb10_Spectral_RBF_SVM_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb10/Spectral' \
# --saveformat='.pdf'

# python analyst.py think \
# --patient='chb10' --method='Spectral' --learner='chb10_Spectral_Logistic_Regression_v2' --train \
# --data='./data' --models='./models' --saveto='./figures/chb10/Spectral' \
# --saveformat='.pdf'
# -------------------------------------------------- 
echo 'analyst.sh terminated'