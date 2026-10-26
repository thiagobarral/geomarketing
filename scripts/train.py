"""Compare models with training-only CV, evaluate once, then predict SP."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
SEED = 7
LABELS = ['Baixo', 'Médio', 'Alto']
FEATURES = [
    'popAte9', 'popDe10a14', 'popDe15a19', 'popDe20a24', 'popDe25a34',
    'popDe35a49', 'popDe50a59', 'popMaisDe60', 'domiciliosA1',
    'domiciliosA2', 'domiciliosB1', 'domiciliosB2', 'domiciliosC1',
    'domiciliosC2', 'domiciliosD', 'domiciliosE', 'rendaMedia',
]


def load_data(path):
    data = pd.read_excel(path)
    required = set(FEATURES + ['codigo', 'nome', 'cidade', 'estado', 'faturamento', 'potencial'])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Colunas ausentes: {sorted(missing)}')
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors='coerce')
    values = data[FEATURES].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError('Preditores devem ser não negativos e não infinitos.')
    rj = data.loc[data.estado.eq('RJ')].copy()
    sp = data.loc[data.estado.eq('SP')].copy()
    if rj.empty or sp.empty:
        raise ValueError('São necessários registros de RJ e SP.')
    if rj[['faturamento', 'potencial']].isna().any().any():
        raise ValueError('Todos os registros de RJ precisam dos dois alvos.')
    if not rj.potencial.isin(LABELS).all():
        raise ValueError('Rótulo de potencial desconhecido.')
    revenue = pd.to_numeric(rj.faturamento, errors='raise')
    if not np.isfinite(revenue).all() or revenue.lt(0).any():
        raise ValueError('Faturamento de RJ inválido.')
    return rj, sp


def pipeline(model):
    # Imputation and scaling are fitted within each CV fold, never on SP/test.
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)


def select_model(candidates, X, y, scoring, cv):
    searches = {}
    rows = []
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(pipeline(model), grid, scoring=scoring, cv=cv,
                              n_jobs=1, error_score='raise')
        search.fit(X, y)
        searches[name] = search
        score = float(search.best_score_)
        rows.append({'modelo': name, 'cv_score': score,
                     'parametros': search.best_params_})
    winner = max(rows, key=lambda row: row['cv_score'])['modelo']
    return winner, searches[winner].best_estimator_, rows


def run(input_path, output_dir):
    rj, sp = load_data(input_path)
    # Same held-out neighborhoods for both tasks; the test is never used to select.
    train_idx, test_idx = train_test_split(
        rj.index, test_size=0.3, random_state=SEED, stratify=rj.potencial)
    train, test = rj.loc[train_idx], rj.loc[test_idx]
    regression = {
        'baseline_media': (DummyRegressor(), {}),
        'linear': (LinearRegression(), {}),
        'svr': (SVR(), {'svr__C': [1.0, 1000.0, 100000.0]}),
        'arvore': (DecisionTreeRegressor(random_state=SEED), {
            'decisiontreeregressor__max_depth': [3, 5, None],
            'decisiontreeregressor__min_samples_leaf': [1, 5, 10]}),
    }
    classification = {
        'baseline_majoritaria': (DummyClassifier(strategy='most_frequent'), {}),
        'logistica': (LogisticRegression(max_iter=2000, random_state=SEED), {}),
        'knn': (KNeighborsClassifier(), {'kneighborsclassifier__n_neighbors': [3, 5, 9]}),
        'svm': (SVC(), {'svc__C': [0.1, 1.0, 10.0]}),
        'random_forest': (RandomForestClassifier(n_estimators=200, random_state=SEED),
                          {'randomforestclassifier__min_samples_leaf': [1, 3, 5]}),
    }
    reg_name, reg_model, reg_rows = select_model(
        regression, train[FEATURES], train.faturamento,
        'neg_root_mean_squared_error', KFold(4, shuffle=True, random_state=SEED))
    cls_name, cls_model, cls_rows = select_model(
        classification, train[FEATURES], train.potencial,
        'f1_macro', StratifiedKFold(4, shuffle=True, random_state=SEED))
    reg_pred = reg_model.predict(test[FEATURES])
    cls_pred = cls_model.predict(test[FEATURES])
    metrics = {
        'seed': SEED, 'n_train': len(train), 'n_test': len(test), 'n_sp': len(sp),
        'regression': {'selected': reg_name, 'cv_candidates': reg_rows,
                       'test_rmse': float(np.sqrt(mean_squared_error(test.faturamento, reg_pred))),
                       'test_r2': float(r2_score(test.faturamento, reg_pred))},
        'classification': {'selected': cls_name, 'cv_candidates': cls_rows,
                           'test_accuracy': float(accuracy_score(test.potencial, cls_pred)),
                           'test_f1_macro': float(f1_score(test.potencial, cls_pred, average='macro')),
                           'labels': LABELS,
                           'confusion_matrix': confusion_matrix(test.potencial, cls_pred, labels=LABELS).tolist(),
                           'report': classification_report(test.potencial, cls_pred, labels=LABELS,
                                                           output_dict=True, zero_division=0)},
    }
    # Refit only after evaluation; SP targets remain unknown.
    final_reg = clone(reg_model).fit(rj[FEATURES], rj.faturamento)
    final_cls = clone(cls_model).fit(rj[FEATURES], rj.potencial)
    result = sp[['codigo', 'nome', 'cidade', 'estado']].copy()
    result['faturamento_estimado'] = np.maximum(0, final_reg.predict(sp[FEATURES]))
    result['potencial_estimado'] = final_cls.predict(sp[FEATURES])
    result['sem_populacao_informada'] = sp['população'].eq(0).to_numpy()
    result['renda_ausente'] = sp.rendaMedia.isna().to_numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    result.to_csv(output_dir / 'previsoes_sp.csv', index=False)
    print(json.dumps({key: {k: v for k, v in metrics[key].items() if k.startswith('test_') or k == 'selected'}
                      for key in ['regression', 'classification']}, ensure_ascii=False, indent=2))
    return metrics, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'DadosDesafioCientista.xlsx')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    run(args.input, args.output_dir)
