import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import seaborn as sns
import time
from tqdm import tqdm
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class Bagging:
    def __init__(self, base_estimator=None, n_estimators=10, bootstrap=True):
        """
        Implementation of Bagging ensemble method
        
        Parameters:
        -----------
        base_estimator : object, default=None
            The base estimator to fit on random subsets of the dataset.
        n_estimators : int, default=10
            The number of base estimators in the ensemble.
        bootstrap : bool, default=True
            Whether samples are drawn with replacement.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.estimators_ = []
        
    def fit(self, X, y):
        """
        Build a Bagging ensemble of estimators.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        
        Returns:
        --------
        self : object
        """
        n_samples = X.shape[0]
        
        # For each base estimator
        for _ in tqdm(range(self.n_estimators), desc="Training Bagging Models"):
            # Create a bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                indices = np.random.choice(n_samples, n_samples, replace=False)
                
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Create and train the base estimator
            estimator = self.base_estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(estimator)
            
        return self
    
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        # Get predictions from all base estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Majority voting
        predictions = np.transpose(predictions)  # Shape: (n_samples, n_estimators)
        
        # For each sample, count occurrences of each class
        y_pred = np.array([np.bincount(pred).argmax() for pred in predictions])
        
        return y_pred

class GradientBoosting:
    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=0.1, subsample=1.0):
        """
        Implementation of Gradient Boosting for classification
        
        Parameters:
        -----------
        base_estimator : object, default=None
            The base estimator to fit on residuals.
        n_estimators : int, default=100
            The number of boosting stages.
        learning_rate : float, default=0.1
            The learning rate shrinks the contribution of each tree.
        subsample : float, default=1.0
            The fraction of samples to be used for fitting the individual base learners.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.estimators_ = []
        self.n_classes = None
        self.feature_importances_ = None
        
    def _encode_target(self, y):
        """One-hot encode the target"""
        self.n_classes = len(np.unique(y))
        y_encoded = np.zeros((len(y), self.n_classes))
        for i in range(len(y)):
            y_encoded[i, y[i]] = 1
        return y_encoded
    
    def fit(self, X, y):
        """
        Build a Gradient Boosting ensemble of estimators.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        
        Returns:
        --------
        self : object
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Encode target as one-hot
        y_encoded = self._encode_target(y)
        
        # Initialize predictions with zeros
        F = np.zeros((X.shape[0], self.n_classes))
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # For each boosting stage
        for _ in tqdm(range(self.n_estimators), desc="Training Gradient Boosting Models"):
            # Subsample the training data
            if self.subsample < 1.0:
                indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_subsample = X[indices]
                y_encoded_subsample = y_encoded[indices]
                F_subsample = F[indices]
            else:
                X_subsample = X
                y_encoded_subsample = y_encoded
                F_subsample = F
            
            # Compute softmax probabilities
            exp_F = np.exp(F_subsample)
            probs = exp_F / np.sum(exp_F, axis=1, keepdims=True)
            
            # Compute negative gradient (residuals)
            residuals = y_encoded_subsample - probs
            
            # Train a base estimator for each class
            stage_estimators = []
            for k in range(self.n_classes):
                estimator = self.base_estimator.__class__(**self.base_estimator.get_params())
                estimator.fit(X_subsample, residuals[:, k])
                stage_estimators.append(estimator)
                
                # Update feature importances if available
                if hasattr(estimator, 'feature_importances_'):
                    self.feature_importances_ += estimator.feature_importances_ / (self.n_estimators * self.n_classes)
            
            self.estimators_.append(stage_estimators)
            
            # Update F for all samples
            for k in range(self.n_classes):
                F[:, k] += self.learning_rate * stage_estimators[k].predict(X)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Initialize predictions with zeros
        F = np.zeros((X.shape[0], self.n_classes))
        
        # For each boosting stage
        for stage_estimators in self.estimators_:
            for k in range(self.n_classes):
                F[:, k] += self.learning_rate * stage_estimators[k].predict(X)
        
        # Compute softmax probabilities
        exp_F = np.exp(F)
        return exp_F / np.sum(exp_F, axis=1, keepdims=True)
    
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        return np.argmax(self.predict_proba(X), axis=1)

def load_and_preprocess_data(dataset_name):
    """
    Load and preprocess the dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load ('mnist' or 'fashion_mnist')
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        The preprocessed training and testing data
    """
    if dataset_name == 'mnist':
        # Load MNIST dataset
        dataset = load_dataset("mnist")
    elif dataset_name == 'fashion_mnist':
        # Load Fashion MNIST dataset
        dataset = load_dataset("fashion_mnist")
    else:
        raise ValueError("Invalid dataset name. Use 'mnist' or 'fashion_mnist'")
    
    # Extract training and testing data
    X_train = np.array([np.array(image) for image in dataset['train']['image']])
    y_train = np.array(dataset['train']['label'])
    X_test = np.array([np.array(image) for image in dataset['test']['image']])
    y_test = np.array(dataset['test']['label'])
    
    # Flatten the images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Add simple statistical features that are computationally efficient
    def add_simple_features(X):
        # Reshape to original image dimensions
        X_images = X.reshape(-1, 28, 28)
        extra_features = []
        
        for img in X_images:
            # Simple row and column means (capture horizontal/vertical distributions)
            row_means = np.mean(img, axis=1)
            col_means = np.mean(img, axis=0)
            
            # Add basic statistical features (5 features total)
            features = np.array([
                np.mean(img),        # Average brightness
                np.std(img),         # Contrast
                np.max(img) - np.min(img)  # Dynamic range
            ])
            
            extra_features.append(np.concatenate([features, row_means[:10], col_means[:10]]))  # Just use first 10 rows/cols
        
        return np.hstack([X, np.array(extra_features)])
    
    # Add simple features (much faster than edge detection)
    X_train = add_simple_features(X_train)
    X_test = add_simple_features(X_test)
    
    # Use a smaller subset of data for training and testing to reduce computation time
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=10000, random_state=42)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=2000, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for printing results
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R2 score
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return {
        'accuracy': acc,
        'mae': mae,
        'r2': r2,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, classes, title, normalize=False):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    classes : list
        List of class labels
    title : str
        Title for the plot
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def run_experiment(dataset_name):
    """
    Run the experiment for a given dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('mnist' or 'fashion_mnist')
    """
    print(f"\n{'='*50}")
    print(f"Running experiment on {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_name)
    
    # Define class names
    if dataset_name == 'mnist':
        class_names = [str(i) for i in range(10)]
    else:  # fashion_mnist
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    
    # 1. Bagging Implementation
    print("\nTraining Bagging model...")
    start_time = time.time()
    
    # Create and train Bagging model with balanced efficiency and accuracy
    bagging_model = Bagging(
        base_estimator=DecisionTreeClassifier(max_depth=8, min_samples_split=5),
        n_estimators=15,
        bootstrap=True
    )
    bagging_model.fit(X_train, y_train)
    
    # Predict using Bagging model
    bagging_pred = bagging_model.predict(X_test)
    
    bagging_time = time.time() - start_time
    print(f"Bagging training time: {bagging_time:.2f} seconds")
    
    # Evaluate Bagging model
    bagging_metrics = evaluate_model(y_test, bagging_pred, "Bagging")
    
    # Plot confusion matrix for Bagging
    plot_confusion_matrix(
        bagging_metrics['confusion_matrix'],
        classes=class_names,
        title=f'Confusion Matrix - Bagging on {dataset_name.upper()}'
    )
    
    # 2. Gradient Boosting Implementation
    print("\nTraining Gradient Boosting model...")
    start_time = time.time()
    
    # Create and train Gradient Boosting model with balanced efficiency and accuracy
    gb_model = GradientBoosting(
        base_estimator=DecisionTreeRegressor(max_depth=2),  # Slightly deeper than stumps
        n_estimators=25,
        learning_rate=0.1,
        subsample=0.8  # Use stochastic gradient boosting
    )
    gb_model.fit(X_train, y_train)
    
    # Predict using Gradient Boosting model
    gb_pred = gb_model.predict(X_test)
    
    gb_time = time.time() - start_time
    print(f"Gradient Boosting training time: {gb_time:.2f} seconds")
    
    # Evaluate Gradient Boosting model
    gb_metrics = evaluate_model(y_test, gb_pred, "Gradient Boosting")
    
    # Plot confusion matrix for Gradient Boosting
    plot_confusion_matrix(
        gb_metrics['confusion_matrix'],
        classes=class_names,
        title=f'Confusion Matrix - Gradient Boosting on {dataset_name.upper()}'
    )
    
    # Compare models
    print("\nModel Comparison:")
    print(f"{'Model':<20} {'Accuracy':<10} {'MAE':<10} {'R2 Score':<10} {'Time (s)':<10}")
    print(f"{'-'*60}")
    print(f"{'Bagging':<20} {bagging_metrics['accuracy']:.4f} {bagging_metrics['mae']:.4f} {bagging_metrics['r2']:.4f} {bagging_time:.2f}")
    print(f"{'Gradient Boosting':<20} {gb_metrics['accuracy']:.4f} {gb_metrics['mae']:.4f} {gb_metrics['r2']:.4f} {gb_time:.2f}")

if __name__ == "__main__":
    # Run experiments on MNIST dataset
    run_experiment('mnist')
    
    # Run experiments on Fashion MNIST dataset
    run_experiment('fashion_mnist')