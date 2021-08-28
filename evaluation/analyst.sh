#!/bin/bash

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


# chb06 training with ARMA
# -------------------------
# python analyst.py teach \
# --patient='chb06' --method='ARMA' --learning_algorithm='Linear SVM' --data='./data' \
# --learnersaveto='./models/chb06/AR'

# python analyst.py teach \
# --patient='chb06' --method='ARMA' --learning_algorithm='RBF SVM' --data='./data' \
# --learnersaveto='./models/chb06/AR'

python analyst.py teach \
--patient='chb06' --method='ARMA' --learning_algorithm='Logistic Regression' --data='./data' \
--learnersaveto='./models/chb06/AR' --plot_figures
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

# python analyst.py think \
# --patient='chb01' --method='ARMA' --learner='./models/chb01/AR/chb01_AR_SVM_Linear.joblib' \
# --data='./data' --saveto='./figures/chb01/AR' \
# --saveformat='.pdf' --debug
