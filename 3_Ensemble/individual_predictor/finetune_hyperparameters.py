import scanpy as sc
import numpy as np
import pandas as pd
import json
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import scvi
import celltypist
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(ref_path, query_path=None, n_hvg=3000):
    ref = sc.read_h5ad(ref_path)
    ref_hvg = ref.copy()
    sc.pp.normalize_total(ref_hvg, target_sum=1e4)
    sc.pp.log1p(ref_hvg)
    sc.pp.highly_variable_genes(ref_hvg, n_top_genes=n_hvg, flavor='cell_ranger', batch_key='patient_id')
    hvg_genes = ref_hvg.var_names[ref_hvg.var['highly_variable']]

    ref_subset = ref[:, hvg_genes].copy()

    if query_path:
        query = sc.read_h5ad(query_path)
        common_genes = hvg_genes.intersection(query.var_names)
        return ref[:, common_genes].copy(), query[:, common_genes].copy()

    return ref_subset, None

def preprocess_standard(adata):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def preprocess_with_scaler(adata, scaler):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    return scaler.transform(X)

def preprocess_log_only(adata):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

def tune_mlp(X, y, groups):
    print("Fine-tuning MLP...")
    param_grid = {
        'hidden_layer_sizes': [(1024, 512, 256), (1024, 512, 256, 128), (2048, 1024, 512)],
        'alpha': [0.0005, 0.001, 0.002]
    }
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    grid = GridSearchCV(mlp, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X, y_enc, groups=groups)
    return grid.best_params_, grid.best_score_, grid.cv_results_

def tune_svm(X, y, groups):
    print("Fine-tuning SVM...")
    param_grid = {
        'C': [5, 10, 15],
        'gamma': ['scale', 0.001, 0.01],
        'kernel': ['rbf']
    }
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    svm = SVC(random_state=42, class_weight='balanced')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X, y_enc, groups=groups)
    return grid.best_params_, grid.best_score_, grid.cv_results_

def tune_xgboost(X, y, groups):
    print("Fine-tuning XGBoost...")
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [7, 8, 9],
        'learning_rate': [0.08, 0.1, 0.12]
    }
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    xgb_clf = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss', tree_method='hist')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    grid = GridSearchCV(xgb_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X, y_enc, groups=groups)
    return grid.best_params_, grid.best_score_, grid.cv_results_

def tune_lightgbm(X, y, groups):
    print("Fine-tuning LightGBM...")
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [7, 8, 9],
        'num_leaves': [15, 20, 25]
    }
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    grid = GridSearchCV(lgb_clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X, y_enc, groups=groups)
    return grid.best_params_, grid.best_score_, grid.cv_results_

def tune_randomforest(X, y, groups):
    print("Fine-tuning RandomForest...")
    param_grid = {
        'n_estimators': [150, 200, 250, 300],
        'max_depth': [12, 15, 18]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X, y, groups=groups)

    best_params = grid.best_params_
    best_params['n_selected_genes'] = 1500
    return best_params, grid.best_score_, grid.cv_results_

def tune_scanvi(ref_adata, groups):
    print("Fine-tuning scANVI with 5-fold CV...")
    configs = [
        {'n_latent': 15, 'n_layers': 2},
        {'n_latent': 20, 'n_layers': 2},
        {'n_latent': 25, 'n_layers': 2},
        {'n_latent': 30, 'n_layers': 2}
    ]

    y = ref_adata.obs['CellType_Merged'].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

    results = []
    for config in configs:
        fold_scores = []
        print(f"  Testing {config}...")

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            ref_train = ref_adata[train_idx].copy()
            ref_val = ref_adata[val_idx].copy()

            ref_train.layers["counts"] = ref_train.X.copy()
            scvi.model.SCVI.setup_anndata(ref_train, layer="counts", labels_key='CellType_Merged', batch_key='patient_id')

            vae = scvi.model.SCVI(ref_train, n_latent=config['n_latent'], n_layers=config['n_layers'])
            vae.train(max_epochs=30, early_stopping=True, accelerator='mps')

            model = scvi.model.SCANVI.from_scvi_model(vae, labels_key='CellType_Merged', unlabeled_category="Unknown")
            model.train(max_epochs=30, early_stopping=True, accelerator='mps')

            ref_val.layers["counts"] = ref_val.X.copy()
            ref_val.obs['true_labels'] = ref_val.obs['CellType_Merged'].copy()
            ref_val.obs['CellType_Merged'] = "Unknown"
            scvi.model.SCANVI.setup_anndata(ref_val, layer="counts", labels_key='CellType_Merged', unlabeled_category="Unknown")

            val_model = scvi.model.SCANVI.load_query_data(ref_val, model)
            val_model.train(max_epochs=30, early_stopping=True, accelerator='mps')

            preds = val_model.predict()
            acc = accuracy_score(ref_val.obs['true_labels'].values, preds)
            fold_scores.append(acc)

        mean_acc = np.mean(fold_scores)
        results.append({**config, 'accuracy': mean_acc})
        print(f"    {config} -> CV Accuracy: {mean_acc:.4f}")

    best = max(results, key=lambda x: x['accuracy'])
    return {k: v for k, v in best.items() if k != 'accuracy'}, best['accuracy'], results

def tune_celltypist(ref_adata, X_log, groups):
    print("Fine-tuning CellTypist...")
    y = ref_adata.obs['CellType_Merged'].values
    param_grid = {
        'C': [0.05, 0.1, 0.2, 0.5],
        'max_iter': [800, 1000, 1200]
    }

    results = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

    for C in param_grid['C']:
        for max_iter in param_grid['max_iter']:
            scores = []
            for train_idx, val_idx in cv.split(X_log, y, groups):
                ref_train = ref_adata[train_idx].copy()
                ref_val = ref_adata[val_idx].copy()

                sc.pp.normalize_total(ref_train, target_sum=1e4)
                sc.pp.log1p(ref_train)
                sc.pp.normalize_total(ref_val, target_sum=1e4)
                sc.pp.log1p(ref_val)

                model = celltypist.train(X=ref_train, labels=y[train_idx], C=C, max_iter=max_iter,
                                        feature_selection=True, balance_cell_type=True, n_jobs=-1, check_expression=True)
                pred = celltypist.annotate(ref_val, model=model, majority_voting=False)
                acc = accuracy_score(y[val_idx], pred.predicted_labels['predicted_labels'].values)
                scores.append(acc)

            mean_acc = np.mean(scores)
            results.append({'C': C, 'max_iter': max_iter, 'accuracy': mean_acc})
            print(f"  C={C}, max_iter={max_iter} -> Accuracy: {mean_acc:.4f}")

    best = max(results, key=lambda x: x['accuracy'])
    return {k: v for k, v in best.items() if k != 'accuracy'}, best['accuracy'], results

def evaluate_on_test(model_name, best_params, ref_train, query_test, X_train_scaled, X_test_scaled, y_train, y_test):
    print(f"Evaluating {model_name} on test set...")

    if model_name == 'MLP':
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = MLPClassifier(**best_params, max_iter=1000, early_stopping=True, random_state=42)
        model.fit(X_train_scaled, y_train_enc)
        y_pred = le.inverse_transform(model.predict(X_test_scaled))

    elif model_name == 'SVM':
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = SVC(**best_params, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train_enc)
        y_pred = le.inverse_transform(model.predict(X_test_scaled))

    elif model_name == 'XGBoost':
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, eval_metric='mlogloss', tree_method='hist')
        model.fit(X_train_scaled, y_train_enc)
        y_pred = le.inverse_transform(model.predict(X_test_scaled))

    elif model_name == 'LightGBM':
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced')
        model.fit(X_train_scaled, y_train_enc)
        y_pred = le.inverse_transform(model.predict(X_test_scaled))

    elif model_name == 'RandomForest':
        rf_params = {k: v for k, v in best_params.items() if k != 'n_selected_genes'}
        model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    elif model_name == 'scANVI':
        ref = ref_train.copy()
        query = query_test.copy()
        ref.layers["counts"] = ref.X.copy()
        scvi.model.SCVI.setup_anndata(ref, layer="counts", labels_key='CellType_Merged')
        vae = scvi.model.SCVI(ref, **best_params)
        vae.train(max_epochs=50, early_stopping=True, accelerator='mps')
        model = scvi.model.SCANVI.from_scvi_model(vae, labels_key='CellType_Merged', unlabeled_category="Unknown")
        model.train(max_epochs=50, early_stopping=True, accelerator='mps')

        query.layers["counts"] = query.X.copy()
        query.obs['CellType_Merged'] = "Unknown"
        scvi.model.SCANVI.setup_anndata(query, layer="counts", labels_key='CellType_Merged', unlabeled_category="Unknown")
        query_model = scvi.model.SCANVI.load_query_data(query, model)
        query_model.train(max_epochs=50, early_stopping=True, accelerator='mps')
        y_pred = query_model.predict()

    elif model_name == 'CellTypist':
        ref = ref_train.copy()
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
        query = query_test.copy()
        sc.pp.normalize_total(query, target_sum=1e4)
        sc.pp.log1p(query)
        model = celltypist.train(X=ref, labels=y_train, **best_params, feature_selection=True, balance_cell_type=True, n_jobs=-1, check_expression=False)
        result = celltypist.annotate(query, model=model, majority_voting=False)
        y_pred = result.predicted_labels['predicted_labels'].values

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"  Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1

if __name__ == "__main__":

    ref, query = load_and_prepare_data('../ref_subset.h5ad', query_path='../query_subset.h5ad')

    y = ref.obs['CellType_Merged'].values
    X_scaled, scaler = preprocess_standard(ref)
    X_log = preprocess_log_only(ref)

    y_test = query.obs['CellType_Merged'].values
    X_test_scaled = preprocess_with_scaler(query, scaler)
    X_test_log = preprocess_log_only(query)

    best_params = {}
    tuning_results = []

    groups = ref.obs['patient_id'].values

    # MLP
    params, score, cv_res = tune_mlp(X_scaled, y, groups)
    best_params['MLP'] = params
    result_row = {'Model': 'MLP', 'CV_Score': score, **params}
    if query is not None:
        test_acc, test_f1 = evaluate_on_test('MLP', params, ref, query, X_scaled, X_test_scaled, y, y_test)
        result_row.update({'Test_Accuracy': test_acc, 'Test_F1': test_f1})
    tuning_results.append(result_row)
    print(f"MLP Best: {params}, CV: {score:.4f}\n")

    # SVM
    params, score, cv_res = tune_svm(X_scaled, y, groups)
    best_params['SVM'] = params
    result_row = {'Model': 'SVM', 'CV_Score': score, **params}
    if query is not None:
        test_acc, test_f1 = evaluate_on_test('SVM', params, ref, query, X_scaled, X_test_scaled, y, y_test)
        result_row.update({'Test_Accuracy': test_acc, 'Test_F1': test_f1})
    tuning_results.append(result_row)
    print(f"SVM Best: {params}, CV: {score:.4f}\n")

    # XGBoost
    params, score, cv_res = tune_xgboost(X_scaled, y, groups)
    best_params['XGBoost'] = params
    result_row = {'Model': 'XGBoost', 'CV_Score': score, **params}
    if query is not None:
        test_acc, test_f1 = evaluate_on_test('XGBoost', params, ref, query, X_scaled, X_test_scaled, y, y_test)
        result_row.update({'Test_Accuracy': test_acc, 'Test_F1': test_f1})
    tuning_results.append(result_row)
    print(f"XGBoost Best: {params}, CV: {score:.4f}\n")

    # LightGBM
    params, score, cv_res = tune_lightgbm(X_scaled, y, groups)
    best_params['LightGBM'] = params
    result_row = {'Model': 'LightGBM', 'CV_Score': score, **params}
    if query is not None:
        test_acc, test_f1 = evaluate_on_test('LightGBM', params, ref, query, X_scaled, X_test_scaled, y, y_test)
        result_row.update({'Test_Accuracy': test_acc, 'Test_F1': test_f1})
    tuning_results.append(result_row)
    print(f"LightGBM Best: {params}, CV: {score:.4f}\n")

    # RandomForest
    params, score, cv_res = tune_randomforest(X_scaled, y, groups)
    best_params['RandomForest'] = params
    result_row = {'Model': 'RandomForest', 'CV_Score': score, **params}
    if query is not None:
        test_acc, test_f1 = evaluate_on_test('RandomForest', params, ref, query, X_scaled, X_test_scaled, y, y_test)
        result_row.update({'Test_Accuracy': test_acc, 'Test_F1': test_f1})
    tuning_results.append(result_row)
    print(f"RandomForest Best: {params}, CV: {score:.4f}\n")

    # scANVI
    params, score, cv_res = tune_scanvi(ref, groups)
    best_params['scANVI'] = params
    result_row = {'Model': 'scANVI', 'CV_Score': score, **params}
    if query is not None:
        test_acc, test_f1 = evaluate_on_test('scANVI', params, ref, query, X_scaled, X_test_scaled, y, y_test)
        result_row.update({'Test_Accuracy': test_acc, 'Test_F1': test_f1})
    tuning_results.append(result_row)
    print(f"scANVI Best: {params}, CV: {score:.4f}\n")

    # CellTypist
    params, score, cv_res = tune_celltypist(ref, X_log, groups)
    best_params['CellTypist'] = params
    result_row = {'Model': 'CellTypist', 'CV_Score': score, **params}
    if query is not None:
        test_acc, test_f1 = evaluate_on_test('CellTypist', params, ref, query, X_scaled, X_test_scaled, y, y_test)
        result_row.update({'Test_Accuracy': test_acc, 'Test_F1': test_f1})
    tuning_results.append(result_row)
    print(f"CellTypist Best: {params}, CV: {score:.4f}\n")

    with open('best_hyperparameters_finetuned.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    df = pd.DataFrame(tuning_results)
    df.to_excel('fine_tuning_results.xlsx', index=False)