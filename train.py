"""Reproducible educational Ames regression benchmark; no external/test labels."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def features(frame):
    x = frame.drop(columns=['Id', 'SalePrice'], errors='ignore').copy()
    for c in ['MSSubClass', 'MoSold']:
        x[c] = x[c].astype(str)
    x['TotalArea'] = x['TotalBsmtSF'].fillna(0) + x['1stFlrSF'] + x['2ndFlrSF']
    x['TotalBath'] = x['FullBath'] + .5*x['HalfBath'] + x['BsmtFullBath'].fillna(0) + .5*x['BsmtHalfBath'].fillna(0)
    x['AgeAtSale'] = x['YrSold'] - x['YearBuilt']
    x['YearsSinceRemodel'] = x['YrSold'] - x['YearRemodAdd']
    # Convert pandas string extension columns to object and missing markers to np.nan.
    for c in x.select_dtypes(include=['object', 'str', 'string']).columns:
        x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
    return x

def models(x):
    num = x.select_dtypes(include='number').columns.tolist()
    cat = [c for c in x if c not in num]
    prep = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), num),
        ('categorical', make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), cat),
    ])
    return {
        'mean_log_baseline': DummyRegressor(strategy='mean'),
        'ridge_alpha20': make_pipeline(clone(prep), Ridge(alpha=20)),
        'gradient_boosting': make_pipeline(clone(prep), GradientBoostingRegressor(n_estimators=650, learning_rate=.035, max_depth=3, min_samples_leaf=8, loss='huber', random_state=42)),
    }

def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data/'train.csv'); test = pd.read_csv(data/'test.csv')
    assert train.Id.is_unique and test.Id.is_unique and not set(train.Id)&set(test.Id)
    assert train.SalePrice.gt(0).all() and 'SalePrice' not in test
    x = features(train); xt = features(test); y = np.log(train.SalePrice.to_numpy())
    folds = list(KFold(5, shuffle=True, random_state=42).split(x))
    results = {}; predictions = {}
    for name, model in models(x).items():
        oof = np.empty(len(x)); scores = []
        for tr, va in folds:
            fitted = clone(model).fit(x.iloc[tr], y[tr]); p = fitted.predict(x.iloc[va])
            oof[va] = p; scores.append(float(root_mean_squared_error(y[va], p)))
        results[name] = {'fold_log_rmse': scores, 'oof_log_rmse': float(root_mean_squared_error(y, oof)), 'dollar_mae': float(mean_absolute_error(train.SalePrice, np.exp(oof)))}
        predictions[name] = oof
        print(name, results[name], flush=True)
    best = min(results, key=lambda k: results[k]['oof_log_rmse'])
    model = models(x)[best]
    # Secondary temporal stress test: fit 2006-2008, test 2009-2010. Not used for model selection.
    tr = train.YrSold.le(2008); va = ~tr
    temporal = clone(model).fit(x.loc[tr], y[tr]).predict(x.loc[va])
    temporal_rmse = float(root_mean_squared_error(y[va], temporal))
    fitted = clone(model).fit(x, y); p = np.exp(fitted.predict(xt))
    assert len(p)==len(test) and np.isfinite(p).all() and (p>0).all()
    submission = pd.DataFrame({'Id':test.Id, 'SalePrice':p})
    assert submission.Id.tolist()==pd.read_csv(data/'sample_submission.csv').Id.tolist()
    submission.to_csv(out/'submission.csv', index=False)
    # Aggregate diagnostics only; raw competition records must not be redistributed.
    diag = pd.DataFrame({'year':train.YrSold,'abs_error':np.abs(train.SalePrice-np.exp(predictions[best]))}).groupby('year').agg(rows=('abs_error','size'), mae=('abs_error','mean'))
    diag.to_csv(out/'oof_error_by_year.csv')
    metrics = {'training_rows':len(train),'test_rows':len(test),'input_features':79,'seed':42,'models':results,'selected_model':best,'temporal_log_rmse':temporal_rmse,'temporal_train_rows':int(tr.sum()),'temporal_test_rows':int(va.sum()),'kaggle_score':None,'data_sha256':{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['train.csv','test.csv']}}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2), flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,default=Path('reports')); a=p.parse_args(); run(a.data,a.out)
