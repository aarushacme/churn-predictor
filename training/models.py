"""
models.py – Model definitions, pipeline packaging, cross-validation, and evaluation.
Ensures preprocessing is bundled with estimators inside complete Pipelines to prevent CV leakage.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


def get_baseline_model():
    """Returns a baseline dummy classifier (predicts the majority class 'No Churn')."""
    return DummyClassifier(strategy='most_frequent')


def get_models():
    """
    Returns the dictionary of primary candidate classifiers.
    Each model choice is academically justified:
    1. Logistic Regression: Linear parametric probabilistic benchmark with direct coefficient interpretability.
    2. Decision Tree: Non-parametric single-tree model capturing non-linear interactions; baseline for tree overfitting.
    3. Random Forest: Bagged ensemble of decorrelated decision trees reducing variance.
    """
    return {
        'Dummy (Majority)':    DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
    }


def build_pipeline(preprocessor, classifier):
    """Wraps preprocessor and classifier in a single scikit-learn Pipeline."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


def cross_validate_candidate(pipeline, X_train, y_train, cv=None):
    """
    Performs 5-fold Stratified Cross-Validation on the raw training dataset.
    Preprocessing is fitted exclusively inside each fold, preventing all leakage.
    
    Returns dictionary of mean and std for F1, Recall, ROC-AUC, Precision, Accuracy.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    scoring = {
        'f1': 'f1',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'accuracy': 'accuracy'
    }
    
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    return {
        'CV_F1_mean': float(scores['test_f1'].mean()),
        'CV_F1_std':  float(scores['test_f1'].std()),
        'CV_Recall_mean': float(scores['test_recall'].mean()),
        'CV_Recall_std':  float(scores['test_recall'].std()),
        'CV_ROC_AUC_mean': float(scores['test_roc_auc'].mean()),
        'CV_ROC_AUC_std':  float(scores['test_roc_auc'].std()),
        'CV_Precision_mean': float(scores['test_precision'].mean()),
        'CV_Accuracy_mean': float(scores['test_accuracy'].mean())
    }


def evaluate_pipeline(pipeline, X_test, y_test, model_name="Model"):
    """
    Evaluates a fitted pipeline on the untouched held-out test set.
    """
    y_pred = pipeline.predict(X_test)
    
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0.5
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': round(float(acc), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1-Score': round(float(f1), 4),
        'ROC-AUC': round(float(auc), 4),
        'Confusion Matrix': cm
    }
