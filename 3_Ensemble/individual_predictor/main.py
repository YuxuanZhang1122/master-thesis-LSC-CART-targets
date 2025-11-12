import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

from scanvi_model import ScANVIModel
from mlp_model import MLPModel
from randomforest_model import RandomForestModel
from xgboost_model import XGBoostModel
from lightgbm_model import LightGBMModel
from svm_model import SVMModel
from celltypist_model import CellTypistModel

def load_and_prepare_data(ref_path, query_path, n_hvg=3000):
    ref = sc.read_h5ad(ref_path)
    query = sc.read_h5ad(query_path)

    ref_hvg = ref.copy()
    sc.pp.normalize_total(ref_hvg, target_sum=1e4)
    sc.pp.log1p(ref_hvg)
    sc.pp.highly_variable_genes(ref_hvg, n_top_genes=n_hvg, flavor='cell_ranger', batch_key='patient_id')

    hvg_genes = ref_hvg.var_names[ref_hvg.var['highly_variable']]
    common_genes = hvg_genes.intersection(query.var_names)

    ref = ref[:, common_genes].copy()
    query = query[:, common_genes].copy()

    return ref, query

def preprocess_standard(adata):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    scaler = StandardScaler()
    adata.X = scaler.fit_transform(X)
    return adata, scaler

def preprocess_query(adata, scaler):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    adata.X = scaler.transform(X)
    return adata

def preprocess_log_only(adata):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    hspc_recall = recall_score(y_true, y_pred, labels=['HSPC'], average='macro', zero_division=0)
    hspc_precision = precision_score(y_true, y_pred, labels=['HSPC'], average='macro', zero_division=0)
    lspc_recall = recall_score(y_true, y_pred, labels=['LSPC'], average='macro', zero_division=0)
    lspc_precision = precision_score(y_true, y_pred, labels=['LSPC'], average='macro', zero_division=0)

    return {
        'Accuracy': acc,
        'F1': f1,
        'HSPC_Recall': hspc_recall,
        'HSPC_Precision': hspc_precision,
        'LSPC_Recall': lspc_recall,
        'LSPC_Precision': lspc_precision
    }

def run_predictor(name, model_class, ref, query, y_true, preprocess_type='scaled'):
    if preprocess_type == 'raw':
        ref_proc = ref.copy()
        query_proc = query.copy()
    elif preprocess_type == 'log_only':
        ref_proc = preprocess_log_only(ref)
        query_proc = preprocess_log_only(query)
    else:
        ref_proc, scaler = preprocess_standard(ref)
        query_proc = preprocess_query(query, scaler)

    model = model_class()
    model.train(ref_proc)
    y_pred = model.predict(query_proc)

    metrics = calc_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm, y_pred

if __name__ == "__main__":
    ref, query = load_and_prepare_data('../ref_subset.h5ad', '../query_subset.h5ad')
    y_true = query.obs['CellType_Merged'].values

    predictors = [
        ('CellTypist', CellTypistModel, 'log_only'),
        ('scANVI', ScANVIModel, 'raw'),
        ('MLP', MLPModel, 'scaled'),
        ('RandomForest', RandomForestModel, 'scaled'),
        ('XGBoost', XGBoostModel, 'scaled'),
        ('LightGBM', LightGBMModel, 'scaled'),
        ('SVM', SVMModel, 'scaled')
    ]

    results = []
    with pd.ExcelWriter('individual_results_finetuned.xlsx') as writer:
        for name, model_class, preprocess_type in predictors:
            print(f"Running {name}...")
            metrics, cm, y_pred = run_predictor(name, model_class, ref, query, y_true, preprocess_type)
            results.append({'Model': name, **metrics})
            print(f"{name} - Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}")

            cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
            cm_df.to_excel(writer, sheet_name=f'{name}_CM')

        df = pd.DataFrame(results)
        df.to_excel(writer, sheet_name='Metrics', index=False)

    print("\nResults saved to individual_results.xlsx")