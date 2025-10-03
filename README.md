# Pulsar Detection and Ensemble Learning Models
## Overview:

This project consists of two major components: a Pulsar Detection Model using Support Vector Machines (SVM) and Ensemble Learning methods for classification on image datasets (MNIST and Fashion MNIST). The goal is to identify pulsar objects in astronomical data and to evaluate the performance of Bagging and Gradient Boosting on image classification tasks.

## 1. Pulsar Detection using SVM:

This part of the project focuses on pulsar detection from a dataset with high class imbalance. The approach involves using Support Vector Machines (SVM) for classification, specifically evaluating multiple kernel functions.

## Key Steps:

### Data Preprocessing:

Dataset consists of training and testing data (pulsar_data_train.csv and pulsar_data_test.csv).

The dataset has missing values which are imputed using mean imputation.

Feature scaling is applied to ensure that all features are on a similar scale.

### SVM Kernel Evaluation:

Different SVM kernels are evaluated: linear, poly, rbf, and sigmoid.

The evaluation metrics include accuracy, precision, recall, F1-score, and AUC (Area Under the Curve).

The Linear kernel is selected based on the best AUC score.

### Hyperparameter Tuning:

The best performing SVM model (with a linear kernel) is fine-tuned using GridSearchCV.

### Feature Importance:

The Skewness of the integrated profile is identified as the most important feature based on the model's coefficients.

### Model Evaluation:

The model is evaluated using the validation dataset and predictions are made for the test set.

The final predictions are saved in pulsar_predictions.csv.

## Key Findings:

Best Kernel: Linear

Best Hyperparameter: C = 0.1

## Final Model Performance:

Accuracy: 98.49%

Recall (Pulsar): 81%

ROC-AUC: 0.969

## 2. Ensemble Learning Models (Bagging vs Gradient Boosting):

This part of the project evaluates and compares the performance of Bagging and Gradient Boosting on two image classification datasets: MNIST and Fashion MNIST.

## Key Steps:

### Data Preprocessing:

Both datasets (MNIST and Fashion MNIST) are preprocessed by flattening images and scaling pixel values to [0, 1].

Additional statistical features are added (mean, standard deviation, dynamic range, etc.) to improve model performance.

### Bagging:

A Bagging ensemble method is implemented using Decision Trees as base estimators.

15 base estimators are used with bootstrap sampling.

### Gradient Boosting:

A Gradient Boosting ensemble method is implemented using Decision Trees as weak learners.

The model is trained with 25 estimators and a learning rate of 0.1.

### Model Evaluation:

Both models are evaluated based on accuracy, MAE (Mean Absolute Error), and R² Score.

The models are trained and evaluated on both MNIST and Fashion MNIST datasets.

### Results:

Bagging outperforms Gradient Boosting in both datasets in terms of accuracy and training time:

MNIST: Bagging achieved 80.9% accuracy, significantly outperforming Gradient Boosting at 72.1%.

Fashion MNIST: Bagging achieved 74.6% accuracy, outperforming Gradient Boosting at 71.2%.

### Key Findings:

Best Model: Bagging (with Decision Trees as base estimators)

Best Performance on MNIST: 80.9% accuracy (Bagging)

Training Time: Bagging is ~4.5 times faster than Gradient Boosting.
