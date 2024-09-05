import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import RegressorChain
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
import os


class XGBOnLinReg(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, precompute=False,
                 copy_X=True, max_iter=1000, tol=0.0001, warm_start=False,
                 positive=False, selection='cyclic',
                 n_estimators=100, max_depth=None, max_leaves=None, learning_rate=0.05,
                 verbosity=0, objective='reg:squarederror', n_jobs=1,
                 subsample=1, colsample_bytree=1, reg_alpha=0, reg_lambda=1,
                 device='cpu',
                 random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.selection = selection
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.objective = objective
        self.n_jobs = n_jobs
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.device = device
        self.random_state = random_state

        self.lin_reg = Lasso(alpha=alpha, fit_intercept=fit_intercept,
                             precompute=precompute, copy_X=copy_X,
                             max_iter=max_iter, tol=tol,
                             warm_start=warm_start, positive=positive,
                             selection=selection, random_state=random_state)

        self.xgb_reg = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                    max_leaves=max_leaves, learning_rate=learning_rate,
                                    verbosity=verbosity, objective=objective,
                                    n_jobs=n_jobs, subsample=subsample,
                                    colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda, device=device,
                                    random_state=random_state)

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y

        self.lin_reg.fit(X, y)
        y_pred = self.lin_reg.predict(X)
        y_resid = y - y_pred

        self.xgb_reg.fit(X, y_resid)

        return self

    def predict(self, X):
        return self.lin_reg.predict(X) + self.xgb_reg.predict(X)
