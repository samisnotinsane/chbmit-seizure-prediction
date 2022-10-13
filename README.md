# Support Vector Machines for Real-Time Seizure Prediction using Neural Data
---

- Relevant source files are available in the [evaluation folder](https://github.com/samisnotinsane/chbmit-seizure-prediction/tree/main/evaluation).

- The main paper can be found in file [Southampton_MSc_Thesis-Signed.pdf](https://github.com/samisnotinsane/chbmit-seizure-prediction/blob/main/Southampton_MSc_Thesis-Signed.pdf)

## Usage

1. Download the CHB-MIT database from https://physionet.org/content/chbmit/1.0.0/ using the command:
`wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/`

2. Hook `data_gen_train_test.py` with `train_test_data.xlsx`. We have already populated the Excel file with relevant information.

3. Uncomment the appropriate command in `analyst.sh` and save it. Then execute it from terminal. You may need to give execute permissions to this script with `chmod`.

4. The program `analyst.py` contains the core algorithms. It has two main methods, `teach` and `think`. You should look at the example commands in `analyst.sh` for guidance on how to invoke these methods.

Notes: We recommend using the Anaconda package manager to install all relevant dependencies required by `analyst.py`.

P.S. Additional notebooks used for experiment during the development of this project has been open sourced and is available on GitHub through the following [link](https://github.com/samisnotinsane/chbmit-seizure-prediction/tree/main/evaluation)

Thank you for your interest in this project.

Best wishes,
Sameen
02.09.2021 04:25 BST
