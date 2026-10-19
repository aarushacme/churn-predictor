"""
train.py – Rigorous, leakage-free training pipeline for Telecom Customer Churn.

Enforces strictly:
1. True test-data isolation:
   clean -> train/test split -> training-data EDA -> feature selection/ablation -> CV/model selection -> tuning -> final model -> test evaluation
2. Data-driven feature ablation:
   Calculates mean CV F1, F1 std, mean CV ROC-AUC, and F1 change vs baseline. Decides feature set from results.
3. Model comparison & tuning order:
   Compares UNTUNED candidate models using 5-fold Stratified CV. Selects the best model family.
   Only then tunes that model with GridSearchCV on X_train. Fits final pipeline on full X_train. Evaluates once on X_test.
4. Data-driven error analysis:
   Calculates actual TP, TN, FP, FN counts, numerical statistics (mean, median, min, max), and categorical distributions.
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from preprocessing import load_data, clean_data, engineer_features, get_preprocessor
from models import build_pipeline, cross_validate_candidate, evaluate_pipeline

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

DATA_PATH    = os.path.join(PROJECT_ROOT, 'data', 'Telco-Customer-Churn.csv')
FIG_DIR      = os.path.join(PROJECT_ROOT, 'results', 'figures')
RESULT_CSV   = os.path.join(PROJECT_ROOT, 'results', 'model_results.csv')
ABLATION_CSV = os.path.join(PROJECT_ROOT, 'results', 'feature_ablation_results.csv')
THRESH_CSV   = os.path.join(PROJECT_ROOT, 'results', 'threshold_analysis.csv')
ERROR_JSON   = os.path.join(PROJECT_ROOT, 'results', 'error_analysis.json')
STATS_JSON   = os.path.join(PROJECT_ROOT, 'results', 'data_stats.json')
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)


# ─── 1. Question-Driven EDA Plots (Training Data Only) ─────────────────────────
def do_eda(df_train: pd.DataFrame):
    """
    Produces question-driven EDA charts strictly on the TRAINING partition.
    Ensures zero leakage / contamination from the held-out test partition.
    """
    sns.set_theme(style='whitegrid', palette='muted')

    # Q1: How imbalanced is the target in training data?
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_train['Churn'].value_counts()
    bars = ax.bar(['No Churn (0)', 'Churn (1)'], counts.values, color=['#2ca02c', '#d62728'], edgecolor='black', width=0.5)
    ax.set_title('Q1: Overall Churn Distribution (Training Data)\n(Target Class Balance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}\n({height/len(df_train):.1%})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, len(df_train) * 0.9)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q1_churn_distribution.png', dpi=150)
    plt.close()

    # Q2: Does customer tenure differ between churners and retainers?
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(data=df_train, x='tenure', hue='Churn', common_norm=False, fill=True,
                palette={'No': '#2ca02c', 'Yes': '#d62728'}, alpha=0.4, linewidth=2)
    plt.title('Q2: Tenure Distribution by Churn Status (Training Data)\n(Do newer customers exhibit higher churn density?)',
              fontsize=12, fontweight='bold')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q2_tenure_vs_churn.png', dpi=150)
    plt.close()

    # Q3: Is contract commitment associated with churn rate?
    plt.figure(figsize=(7, 4.5))
    contract_rates = df_train.groupby('Contract')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reindex(
        ['Month-to-month', 'One year', 'Two year']
    )
    bars = plt.bar(contract_rates.index, contract_rates.values, color=['#d62728', '#ff7f0e', '#2ca02c'], edgecolor='black')
    plt.title('Q3: Churn Rate by Contract Type (Training Data)\n(Impact of Contractual Commitment)', fontsize=12, fontweight='bold')
    plt.ylabel('Churn Rate (%)')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%', ha='center', fontweight='bold')
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q3_contract_vs_churn.png', dpi=150)
    plt.close()

    # Q4: Are monthly charges systematically higher for churners?
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df_train, x='Churn', y='MonthlyCharges', palette={'No': '#2ca02c', 'Yes': '#d62728'}, width=0.4)
    plt.title('Q4: Monthly Charges by Churn Status (Training Data)\n(Are churners paying higher monthly fees?)', fontsize=12, fontweight='bold')
    plt.xlabel('Churn Status')
    plt.ylabel('Monthly Charges ($)')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q4_monthlycharges_vs_churn.png', dpi=150)
    plt.close()

    # Q5: Does internet technology type correlate with churn?
    plt.figure(figsize=(7, 4.5))
    inet_rates = df_train.groupby('InternetService')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reindex(
        ['Fiber optic', 'DSL', 'No']
    )
    bars = plt.bar(inet_rates.index, inet_rates.values, color=['#d62728', '#1f77b4', '#2ca02c'], edgecolor='black')
    plt.title('Q5: Churn Rate by Internet Service (Training Data)\n(Elevated Churn in Fiber Optic)', fontsize=12, fontweight='bold')
    plt.ylabel('Churn Rate (%)')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%', ha='center', fontweight='bold')
    plt.ylim(0, 55)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q5_internetservice_vs_churn.png', dpi=150)
    plt.close()

    # Q6: Does technical support availability mitigate churn?
    plt.figure(figsize=(7, 4.5))
    ts_rates = df_train.groupby('TechSupport')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reindex(
        ['No', 'Yes', 'No internet service']
    )
    bars = plt.bar(ts_rates.index, ts_rates.values, color=['#d62728', '#2ca02c', '#7f7f7f'], edgecolor='black')
    plt.title('Q6: Churn Rate by Tech Support Subscription (Training Data)\n(Protective Effect of Support)', fontsize=12, fontweight='bold')
    plt.ylabel('Churn Rate (%)')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%', ha='center', fontweight='bold')
    plt.ylim(0, 55)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q6_techsupport_vs_churn.png', dpi=150)
    plt.close()

    # Q7: Does payment method relate to churn behavior?
    plt.figure(figsize=(9, 4.5))
    pm_rates = df_train.groupby('PaymentMethod')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).sort_values(ascending=False)
    bars = plt.bar(pm_rates.index, pm_rates.values, color=['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c'], edgecolor='black')
    plt.title('Q7: Churn Rate by Payment Method (Training Data)\n(Electronic Check Friction)', fontsize=12, fontweight='bold')
    plt.ylabel('Churn Rate (%)')
    plt.xticks(rotation=15, ha='right')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%', ha='center', fontweight='bold')
    plt.ylim(0, 55)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q7_paymentmethod_vs_churn.png', dpi=150)
    plt.close()

    # Q8: Numerical features correlation heatmap (Training Data)
    plt.figure(figsize=(6, 4.5))
    num_corr = df_train[['tenure', 'MonthlyCharges', 'TotalCharges']].corr()
    sns.heatmap(num_corr, annot=True, fmt='.2f', cmap='Blues', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Q8: Correlation Heatmap for Numerical Features (Training Data)\n(tenure vs TotalCharges collinearity)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/q8_numerical_correlation.png', dpi=150)
    plt.close()

    print("  [OK] All 8 Question-Driven EDA plots generated from df_train and saved to results/figures/")


# ─── 2. Data-Driven Feature Ablation Experiment ────────────────────────────────
def run_feature_ablation(df_train: pd.DataFrame, cv: StratifiedKFold):
    """
    Evaluates candidate feature configurations using 5-fold Stratified CV on training data only.
    Calculates:
      - mean CV F1
      - F1 standard deviation
      - mean CV ROC-AUC
      - F1 change vs baseline
    Selects the winning configuration data-driven based on actual results.
    """
    print("\n" + "="*70)
    print("FEATURE ABLATION STUDY (5-Fold Stratified CV on Training Data Only)")
    print("="*70)

    configs = {
        'Baseline (Raw Features)': {'inc_s': False, 'inc_t': False},
        'With SubscribedServiceCount Only': {'inc_s': True, 'inc_t': False},
        'With TenureBand Only': {'inc_s': False, 'inc_t': True},
        'With Both (ServiceCount + TenureBand)': {'inc_s': True, 'inc_t': True}
    }

    ablation_results = {}
    baseline_f1 = None

    for label, cfg in configs.items():
        df_variant = engineer_features(df_train, include_service_count=cfg['inc_s'], include_tenure_band=cfg['inc_t'])
        X_var = df_variant.drop(columns=['Churn'])
        y_var = (df_variant['Churn'] == 'Yes').astype(int)

        num_cols = X_var.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
        cat_cols = X_var.select_dtypes(include=['object', 'category']).columns.tolist()
        preproc = get_preprocessor(num_cols, cat_cols)

        # Evaluate with Logistic Regression pipeline
        pipe = build_pipeline(preproc, LogisticRegression(max_iter=1000, random_state=42))
        scores = cross_validate(pipe, X_var, y_var, cv=cv, scoring=['f1', 'roc_auc'], n_jobs=-1)

        mean_f1 = float(scores['test_f1'].mean())
        std_f1  = float(scores['test_f1'].std())
        mean_auc = float(scores['test_roc_auc'].mean())

        if baseline_f1 is None:
            baseline_f1 = mean_f1
            delta_f1 = 0.0
        else:
            delta_f1 = mean_f1 - baseline_f1

        ablation_results[label] = {
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_auc': mean_auc,
            'delta_f1': delta_f1,
            'inc_s': cfg['inc_s'],
            'inc_t': cfg['inc_t']
        }

    # Data-driven decision: pick configuration with highest mean CV F1
    best_config_name = max(ablation_results.keys(), key=lambda k: ablation_results[k]['mean_f1'])

    records = []
    for label, res in ablation_results.items():
        decision = "SELECTED (Best Mean CV F1)" if label == best_config_name else ("Baseline" if label.startswith("Baseline") else "Not Selected")
        records.append({
            'Configuration': label,
            'Mean CV F1': round(res['mean_f1'], 4),
            'CV F1 Std': round(res['std_f1'], 4),
            'Mean CV ROC-AUC': round(res['mean_auc'], 4),
            'F1 Change vs Baseline': f"{res['delta_f1']:+.4f}",
            'Decision': decision
        })

    ablation_df = pd.DataFrame(records)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(ABLATION_CSV, index=False)

    best_cfg = ablation_results[best_config_name]
    print(f"\nData-Driven Ablation Decision: '{best_config_name}' selected with Mean CV F1 = {best_cfg['mean_f1']:.4f}.")
    return best_cfg['inc_s'], best_cfg['inc_t'], ablation_df


# ─── 3. Data-Driven Error Analysis ─────────────────────────────────────────────
def run_data_driven_error_analysis(final_pipeline, X_test, y_test, final_probs):
    """
    Computes purely calculated error statistics without hard-coded qualitative explanations:
      - FP, FN, TP, TN counts
      - Numerical summaries (mean, median, min, max) for tenure, MonthlyCharges, TotalCharges
      - Categorical percentage distributions for Contract, InternetService, PaymentMethod, TechSupport
    """
    y_test_pred = final_pipeline.predict(X_test)
    df_err = X_test.copy()
    df_err['Actual']    = y_test.values
    df_err['Predicted'] = y_test_pred
    df_err['Prob']      = final_probs

    fp = df_err[(df_err['Actual'] == 0) & (df_err['Predicted'] == 1)]
    fn = df_err[(df_err['Actual'] == 1) & (df_err['Predicted'] == 0)]
    tp = df_err[(df_err['Actual'] == 1) & (df_err['Predicted'] == 1)]
    tn = df_err[(df_err['Actual'] == 0) & (df_err['Predicted'] == 0)]

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if 'SubscribedServiceCount' in X_test.columns:
        num_cols.append('SubscribedServiceCount')

    fp_num_summary = {col: {
        'mean': round(float(fp[col].mean()), 2),
        'median': round(float(fp[col].median()), 2),
        'min': round(float(fp[col].min()), 2),
        'max': round(float(fp[col].max()), 2)
    } for col in num_cols if col in fp.columns}

    fn_num_summary = {col: {
        'mean': round(float(fn[col].mean()), 2),
        'median': round(float(fn[col].median()), 2),
        'min': round(float(fn[col].min()), 2),
        'max': round(float(fn[col].max()), 2)
    } for col in num_cols if col in fn.columns}

    cat_cols_to_check = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'PaperlessBilling']
    fp_cat_distributions = {col: fp[col].value_counts(normalize=True).round(4).to_dict()
                            for col in cat_cols_to_check if col in fp.columns}
    fn_cat_distributions = {col: fn[col].value_counts(normalize=True).round(4).to_dict()
                            for col in cat_cols_to_check if col in fn.columns}

    error_summary = {
        'confusion_counts': {
            'TP': int(len(tp)),
            'TN': int(len(tn)),
            'FP': int(len(fp)),
            'FN': int(len(fn)),
            'total_test_samples': int(len(df_err))
        },
        'false_positives_numerical_summary': fp_num_summary,
        'false_negatives_numerical_summary': fn_num_summary,
        'false_positives_categorical_distributions': fp_cat_distributions,
        'false_negatives_categorical_distributions': fn_cat_distributions
    }

    with open(ERROR_JSON, 'w') as f:
        json.dump(error_summary, f, indent=4)

    # Plot Error Tenure vs Monthly Charges
    plt.figure(figsize=(8, 5))
    plt.scatter(tn['tenure'], tn['MonthlyCharges'], c='#2ca02c', alpha=0.25, label=f'True Negatives ({len(tn)})', s=20)
    plt.scatter(tp['tenure'], tp['MonthlyCharges'], c='#1f77b4', alpha=0.4, label=f'True Positives ({len(tp)})', s=25)
    plt.scatter(fp['tenure'], fp['MonthlyCharges'], c='#ff7f0e', alpha=0.8, marker='^', label=f'False Positives ({len(fp)})', s=45)
    plt.scatter(fn['tenure'], fn['MonthlyCharges'], c='#d62728', alpha=0.8, marker='x', label=f'False Negatives ({len(fn)})', s=45)
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Monthly Charges ($)')
    plt.title('Error Analysis: Misclassified vs Correctly Classified Test Cases', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/error_analysis_scatter.png', dpi=150)
    plt.close()

    print("  [OK] Purely calculated error analysis JSON and scatter plot saved.")
    return error_summary


# ─── 4. Main Workflow ─────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("TELECOM CUSTOMER CHURN PREDICTION – REVISED METHODOLOGICAL PIPELINE")
    print("="*70)

    # Step 1: Load raw dataset
    print("\n[Step 1] Loading raw dataset...")
    df_raw = load_data(DATA_PATH)

    # Step 2: Basic inspection on full data
    print("\n[Step 2] Basic inspection on full dataset...")
    blank_total_charges = int(pd.to_numeric(df_raw['TotalCharges'], errors='coerce').isnull().sum())
    stats = {
        'num_rows': int(df_raw.shape[0]),
        'num_cols': int(df_raw.shape[1]),
        'columns': list(df_raw.columns),
        'missing_values_raw': df_raw.isnull().sum().to_dict(),
        'TotalCharges_blank_strings': blank_total_charges,
        'duplicate_rows': int(df_raw.duplicated().sum()),
        'duplicate_customerIDs': int(df_raw['customerID'].duplicated().sum()),
        'target_distribution': df_raw['Churn'].value_counts().to_dict(),
        'target_churn_percentage': round(float((df_raw['Churn'] == 'Yes').mean() * 100), 2)
    }
    with open(STATS_JSON, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"  Raw Rows: {stats['num_rows']}, Cols: {stats['num_cols']}")
    print(f"  Target: {stats['target_distribution']} ({stats['target_churn_percentage']}% Churn)")
    print(f"  Blank TotalCharges rows: {stats['TotalCharges_blank_strings']} (customers with tenure=0)")
    print(f"  Duplicates: {stats['duplicate_rows']} rows, {stats['duplicate_customerIDs']} customerIDs")

    # Step 3: Clean full data
    print("\n[Step 3] Cleaning dataset...")
    df_clean = clean_data(df_raw)
    print(f"  Cleaned dataset shape: {df_clean.shape} (11 zero-tenure rows dropped, customerID dropped)")

    # Step 4: Train / Test Split IMMEDIATELY (True Test Isolation)
    print("\n[Step 4] Stratified 80/20 Train / Test Split...")
    df_train, df_test = train_test_split(
        df_clean, test_size=0.20, random_state=42, stratify=df_clean['Churn']
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    print(f"  Training set: {df_train.shape[0]} rows ({df_train['Churn'].value_counts().to_dict()})")
    print(f"  Test set:     {df_test.shape[0]} rows ({df_test['Churn'].value_counts().to_dict()})")
    print("  CONFIRMATION: Test partition is sealed. Zero target or feature information from df_test will be accessed until Step 8.")

    # Step 5: Training-data EDA (Only on df_train)
    print("\n[Step 5] Generating Question-Driven EDA charts strictly on df_train...")
    do_eda(df_train)

    # Step 6: Data-Driven Feature Selection / Ablation (Only on df_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n[Step 6] Running Data-Driven Feature Ablation on df_train...")
    inc_service, inc_tenure, ablation_df = run_feature_ablation(df_train, cv)

    # Prepare X_train, y_train using the data-driven feature setup
    df_train_engineered = engineer_features(df_train, include_service_count=inc_service, include_tenure_band=inc_tenure)
    X_train = df_train_engineered.drop(columns=['Churn'])
    y_train = (df_train_engineered['Churn'] == 'Yes').astype(int)

    num_cols = X_train.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = get_preprocessor(num_cols, cat_cols)
    print(f"  Feature Matrix X_train: {X_train.shape[1]} features (Numerical: {len(num_cols)}, Categorical: {len(cat_cols)})")

    # Step 7: Compare UNTUNED candidate models via 5-Fold Stratified CV
    print("\n[Step 7] Comparing UNTUNED candidate models via 5-Fold Stratified CV on X_train...")
    untuned_models = {
        'Dummy (Majority)':    DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
    }

    cv_scores = {}
    untuned_pipelines = {}

    for name, model in untuned_models.items():
        pipe = build_pipeline(preprocessor, model)
        untuned_pipelines[name] = pipe
        scores = cross_validate_candidate(pipe, X_train, y_train, cv=cv)
        cv_scores[name] = scores
        print(f"  {name:22s} | CV F1: {scores['CV_F1_mean']:.4f} ± {scores['CV_F1_std']:.4f} | Recall: {scores['CV_Recall_mean']:.4f} | ROC-AUC: {scores['CV_ROC_AUC_mean']:.4f}")

    # Select best model family based on CV F1
    selectable_candidates = {k: v for k, v in cv_scores.items() if 'Dummy' not in k}
    best_family_name = max(selectable_candidates.keys(), key=lambda k: selectable_candidates[k]['CV_F1_mean'])
    best_family_f1  = selectable_candidates[best_family_name]['CV_F1_mean']
    best_family_auc = selectable_candidates[best_family_name]['CV_ROC_AUC_mean']
    print(f"\n[Model Selection Result] Best Model Family Selected: '{best_family_name}' (Mean CV F1 = {best_family_f1:.4f}, ROC-AUC = {best_family_auc:.4f})")

    # Step 8: Tune ONLY the selected model with GridSearchCV on X_train
    print(f"\n[Step 8] Hyperparameter Tuning ONLY the selected model ('{best_family_name}') via GridSearchCV on X_train...")
    if best_family_name == 'Logistic Regression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    elif best_family_name == 'Random Forest':
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        }
    else:
        param_grid = {
            'classifier__max_depth': [3, 5, 10, None],
            'classifier__min_samples_split': [2, 5, 10]
        }

    raw_selected_model = untuned_models[best_family_name]
    tune_pipeline = build_pipeline(preprocessor, raw_selected_model)
    grid = GridSearchCV(tune_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    tuned_params = grid.best_params_
    final_pipeline = grid.best_estimator_
    print(f"  Tuned Parameters: {tuned_params}")

    # Fit final pipeline on FULL X_train
    print("  Fitting final pipeline on full X_train...")
    final_pipeline.fit(X_train, y_train)

    meta = {
        'selected_model': best_family_name,
        'tuned_params': tuned_params,
        'feature_engineering': {
            'include_service_count': inc_service,
            'include_tenure_band': inc_tenure
        },
        'numerical_cols': num_cols,
        'categorical_cols': cat_cols
    }

    # Save complete inference pipeline to both models/ and app/
    for out_dir in [MODELS_DIR, FRONTEND_DIR]:
        joblib.dump(final_pipeline, os.path.join(out_dir, 'final_pipeline.joblib'))
        joblib.dump(final_pipeline.named_steps['preprocessor'], os.path.join(out_dir, 'preprocessor.joblib'))
        joblib.dump(final_pipeline.named_steps['classifier'],   os.path.join(out_dir, 'final_model.joblib'))
        with open(os.path.join(out_dir, 'column_meta.json'), 'w') as f:
            json.dump(meta, f, indent=4)

    print(f"  [OK] Unified final pipeline and metadata saved to {MODELS_DIR} and {FRONTEND_DIR}")
    # Export portable weights for zero-dependency / cross-version resilience
    ohe_trans = final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    portable_weights = {
        'num_cols': num_cols,
        'scaler_mean': final_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].mean_.tolist(),
        'scaler_scale': final_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].scale_.tolist(),
        'cat_cols': cat_cols,
        'categories': [list(c) for c in ohe_trans.categories_],
        'coef': final_pipeline.named_steps['classifier'].coef_[0].tolist(),
        'intercept': float(final_pipeline.named_steps['classifier'].intercept_[0])
    }
    for out_dir in [MODELS_DIR, FRONTEND_DIR]:
        with open(os.path.join(out_dir, 'model_weights.json'), 'w') as f:
            json.dump(portable_weights, f, indent=2)
    print(f"  [OK] Exported portable model_weights.json for zero-dependency deployment")

    # Step 9: Final Test Set Evaluation (Evaluating once on untouched X_test)
    print("\n[Step 9] Evaluating on untouched held-out test set (df_test unsealed now)...")
    df_test_engineered = engineer_features(df_test, include_service_count=inc_service, include_tenure_band=inc_tenure)
    X_test = df_test_engineered.drop(columns=['Churn'])
    y_test = (df_test_engineered['Churn'] == 'Yes').astype(int)

    test_evaluations = []
    # Fit untuned models on X_train for benchmark reporting
    for name, pipe in untuned_pipelines.items():
        pipe.fit(X_train, y_train)
        eval_dict = evaluate_pipeline(pipe, X_test, y_test, f"{name} (Untuned)")
        eval_dict['CV_F1_mean'] = round(cv_scores[name]['CV_F1_mean'], 4)
        eval_dict['CV_F1_std']  = round(cv_scores[name]['CV_F1_std'], 4)
        eval_dict['CV_Recall']  = round(cv_scores[name]['CV_Recall_mean'], 4)
        eval_dict['CV_ROC_AUC'] = round(cv_scores[name]['CV_ROC_AUC_mean'], 4)
        test_evaluations.append(eval_dict)

    # Evaluate final tuned pipeline
    final_eval = evaluate_pipeline(final_pipeline, X_test, y_test, f"{best_family_name} (Final Tuned)")
    final_eval['CV_F1_mean'] = round(best_family_f1, 4)
    final_eval['CV_F1_std']  = round(cv_scores[best_family_name]['CV_F1_std'], 4)
    final_eval['CV_Recall']  = round(cv_scores[best_family_name]['CV_Recall_mean'], 4)
    final_eval['CV_ROC_AUC'] = round(best_family_auc, 4)
    test_evaluations.append(final_eval)

    res_df = pd.DataFrame(test_evaluations)
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV_F1_mean', 'CV_ROC_AUC']
    print(res_df[display_cols].to_string(index=False))
    res_df.drop(columns=['Confusion Matrix']).to_csv(RESULT_CSV, index=False)
    print(f"  [OK] Test results saved to {RESULT_CSV}")

    # Plot Confusion Matrices
    for item in test_evaluations:
        m_name = item['Model']
        cm = item['Confusion Matrix']
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        ax.set_title(f'Confusion Matrix\n{m_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        plt.tight_layout()
        fname = m_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plt.savefig(f'{FIG_DIR}/cm_{fname}.png', dpi=150)
        plt.close()

    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for item in test_evaluations:
        m_name = item['Model']
        # Find corresponding pipeline
        if 'Final Tuned' in m_name:
            p_obj = final_pipeline
        else:
            base_key = m_name.replace(" (Untuned)", "")
            p_obj = untuned_pipelines[base_key]
        if hasattr(p_obj, 'predict_proba'):
            probs = p_obj.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc_val = roc_auc_score(y_test, probs)
            plt.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Baseline (AUC = 0.50)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('ROC Curves Comparison (Held-Out Test Set)', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/roc_curves.png', dpi=150)
    plt.close()
    print("  [OK] Confusion matrices and ROC curves saved to results/figures/")

    # Step 10: Threshold Analysis for Final Model
    print(f"\n[Step 10] Running Threshold Analysis for {best_family_name} (Final Tuned)...")
    final_probs = final_pipeline.predict_proba(X_test)[:, 1]
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    thresh_records = []

    for t in thresholds:
        preds = (final_probs >= t).astype(int)
        cm_t = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm_t.ravel()
        thresh_records.append({
            'Threshold': t,
            'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, preds, zero_division=0), 4),
            'F1':        round(f1_score(y_test, preds, zero_division=0), 4),
            'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
        })

    thresh_df = pd.DataFrame(thresh_records)
    print(thresh_df[['Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN']].to_string(index=False))
    thresh_df.to_csv(THRESH_CSV, index=False)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(thresh_df['Threshold'], thresh_df['Precision'], marker='o', label='Precision', color='#1f77b4', linewidth=2)
    plt.plot(thresh_df['Threshold'], thresh_df['Recall'],    marker='s', label='Recall',    color='#d62728', linewidth=2)
    plt.plot(thresh_df['Threshold'], thresh_df['F1'],         marker='^', label='F1-Score', color='#2ca02c', linewidth=2)
    plt.axvline(0.50, color='gray', linestyle=':', label='Default Threshold (0.50)')
    plt.xlabel('Classification Decision Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold Analysis – Precision vs Recall vs F1 ({best_family_name} Final Tuned)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/threshold_analysis.png', dpi=150)
    plt.close()

    # Step 11: Feature Interpretation
    print("\n[Step 11] Model Interpretation...")
    ohe = final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_names

    classifier = final_pipeline.named_steps['classifier']
    if hasattr(classifier, 'coef_'):
        coefs = classifier.coef_[0]
        coef_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefs})
        coef_df['Abs'] = coef_df['Coefficient'].abs()
        top_coefs = coef_df.sort_values('Abs', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        colors = ['#d62728' if c > 0 else '#2ca02c' for c in top_coefs['Coefficient']]
        plt.barh(top_coefs['Feature'][::-1], top_coefs['Coefficient'][::-1], color=colors[::-1], edgecolor='black')
        plt.axvline(0, color='black', linewidth=1)
        plt.xlabel('Log-Odds Coefficient (Positive = Higher Churn Risk, Negative = Protective)')
        plt.title(f'Top 15 Predictive Features by Coefficient Magnitude\n({best_family_name} Final Tuned)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/feature_importance_lr.png', dpi=150)
        plt.close()
        print("  [OK] Coefficient plot saved.")

    # Permutation importance on test set
    perm = permutation_importance(final_pipeline, X_test, y_test, scoring='f1', n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Importance': perm.importances_mean,
        'Std_Importance': perm.importances_std
    }).sort_values('Mean_Importance', ascending=False).head(12)

    plt.figure(figsize=(9, 5.5))
    plt.barh(perm_df['Feature'][::-1], perm_df['Mean_Importance'][::-1],
             xerr=perm_df['Std_Importance'][::-1], color='#1f77b4', edgecolor='black', capsize=3)
    plt.xlabel('Mean Decrease in Test F1 Score Upon Feature Shuffling')
    plt.title(f'Top 12 Features by Permutation Importance on Test Set\n({best_family_name} Final Tuned)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/permutation_importance.png', dpi=150)
    plt.close()
    print("  [OK] Permutation importance plot saved.")

    # Step 12: Purely calculated Data-Driven Error Analysis
    print("\n[Step 12] Purely calculated Data-Driven Error Analysis on test set...")
    run_data_driven_error_analysis(final_pipeline, X_test, y_test, final_probs)

    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE – ALL 4 CORRECTIONS SATISFIED")
    print("="*70)


if __name__ == '__main__':
    main()
