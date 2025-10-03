import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Set display options for better output
pd.set_option('display.max_columns', None)
np.set_printoptions(precision=3, suppress=True)

# 2. Data Preparation
# Import the training dataset
train_data = pd.read_csv('pulsar_data_train.csv')
test_data = pd.read_csv('pulsar_data_test.csv')

# Basic exploratory data analysis (EDA)
print("Training Dataset Shape:", train_data.shape)
print("\nTraining Dataset Preview:")
print(train_data.head())

print("\nTraining Dataset Summary Statistics:")
print(train_data.describe())

print("\nTraining Dataset Class Distribution:")
print(train_data['target_class'].value_counts())
print("Class Distribution Percentage:")
print(train_data['target_class'].value_counts(normalize=True) * 100)

# Check for missing values
print("\nMissing Values in Training Data:")
print(train_data.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_data.isnull().sum())

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target_class', data=train_data)
plt.title('Class Distribution')
plt.xlabel('Target Class (0: Non-Pulsar, 1: Pulsar)')
plt.ylabel('Count')
plt.show()

# Examine feature correlations
plt.figure(figsize=(12, 10))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Pairplot for feature relationships
plt.figure(figsize=(12, 10))
feature_cols = train_data.columns[:-1]
sns.pairplot(train_data, vars=feature_cols, hue='target_class', diag_kind='kde')
plt.suptitle('Pairplot of Features by Class', y=1.02)
plt.show()

# Define features and target variable
X_train = train_data.drop('target_class', axis=1)
y_train = train_data['target_class']
X_test = test_data.drop('target_class', axis=1)

# Get feature names for later use
feature_names = X_train.columns

# Check for and handle NaN values
print("\nChecking for NaN values before model building:")
print("NaN values in X_train:", X_train.isna().sum().sum())
print("NaN values in X_test:", X_test.isna().sum().sum())

# 3. Model Building
# Create a function to evaluate different SVM kernels
def evaluate_svm_kernel(kernel_name, X_train, y_train, X_test, y_test):
    """Train an SVM with specified kernel and evaluate its performance"""
    # Create a pipeline with imputation, standard scaling and SVM
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel_name, random_state=42, probability=True))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pipeline.fit(X_train_cv, y_train_cv)
        cv_scores.append(pipeline.score(X_val_cv, y_val_cv))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for SVM with {kernel_name} Kernel')
    plt.legend(loc='lower right')
    plt.show()
    
    # Return results
    return {
        'kernel': kernel_name,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'auc_score': auc_score,
        'cv_scores': cv_scores,
        'cv_mean_accuracy': np.mean(cv_scores),
        'model': pipeline
    }

# Split data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("\n===== Evaluating Different SVM Kernels =====")
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    print(f"\n----- SVM with {kernel} kernel -----")
    result = evaluate_svm_kernel(kernel, X_train_split, y_train_split, X_val, y_val)
    results[kernel] = result
    
    print(f"Confusion Matrix:\n{result['confusion_matrix']}")
    print(f"\nClassification Report:\n{result['classification_report']}")
    print(f"ROC-AUC Score: {result['auc_score']:.4f}")
    print(f"Cross-Validation Scores: {result['cv_scores']}")
    print(f"Mean CV Accuracy: {result['cv_mean_accuracy']:.4f}")

# 4. Find the best kernel based on AUC score
best_kernel = max(results, key=lambda k: results[k]['auc_score'])
print(f"\nBest kernel based on AUC score: {best_kernel}")

# 5. Hyperparameter Tuning for the best kernel
print(f"\n===== Hyperparameter Tuning for SVM with {best_kernel} kernel =====")

# Define parameter grid based on the best kernel
if best_kernel == 'linear':
    param_grid = {
        'svm__C': [0.1, 1, 10, 100]
    }
elif best_kernel == 'poly':
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__degree': [2, 3, 4],
        'svm__gamma': ['scale', 'auto', 0.1, 1]
    }
elif best_kernel == 'rbf':
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
elif best_kernel == 'sigmoid':
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

# Create pipeline for grid search
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel=best_kernel, probability=True, random_state=42))
])

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Print best parameters and results
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate the best model on the validation set
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
y_val_prob = best_model.predict_proba(X_val)[:, 1]

print("\nBest Model Evaluation on Validation Set:")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_val, y_val_prob):.4f}")

# Feature importance analysis (if linear kernel)
if best_kernel == 'linear':
    # Get coefficients
    coefficients = best_model.named_steps['svm'].coef_[0]
    # Create a dataframe for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients)
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Absolute Coefficient Values)')
    plt.tight_layout()
    plt.show()
    
    print("\nFeature Importance:")
    print(feature_importance)

# 6. Make predictions on the test set
final_predictions = best_model.predict(X_test)
final_probabilities = best_model.predict_proba(X_test)[:, 1]

# Create submission dataframe
submission = pd.DataFrame({
    'Index': range(len(final_predictions)),
    'Predicted_Class': final_predictions,
    'Pulsar_Probability': final_probabilities
})

print("\nTest Set Predictions (First 10 rows):")
print(submission.head(10))

# Save the predictions to CSV
submission.to_csv('pulsar_predictions.csv', index=False)
print("\nPredictions saved to 'pulsar_predictions.csv'")

