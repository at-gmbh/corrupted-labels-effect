# imports
import json
import os
import re
import time
from tabnanny import verbose

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, Ridge, SGDRegressor)
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR


def hyperparameter_tuning(
        estimator_name, X_train, y_train, X_test, y_test, regr_result_log, cv=10
    ):
        """
        Tune hyperparameters of estimator using grid search and cross_val_score.
    
        Parameters
        ----------
        estimator_name : str
            Class name of estimator to be tuned
        X_train : numpy.ndarray
            Training data regressor
        y_train : numpy.ndarray
            Training data regressand
        X_test : numpy.ndarray
            Test data regressor
        y_test : numpy.ndarray
            Test data regressand
        regr_result_log : dict{str: dict{str: any}}
            Dictionary with regression propertires for current regression
        cv : int
            Number of cross-validation folds

        Returns
        -------
        regr_result_log : dict{str: dict{str: any}}
            Dictionary with regression propertires for current regression
            including best estimator results
        """
        # load estimator and set parameters
        estimator, params = init_estimator(estimator_name)

        # run grid search
        start = time.time()
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        gs.fit(X_train, y_train)

        # get r^2 score
        r2_train = r2_score(y_train, gs.predict(X_train))
        r2_test = r2_score(y_test, gs.predict(X_test))

        end = time.time()

        print(
            f'\t{estimator}'\
            f' - Training time elapsed: {round((end - start) / 60, 3)} min'\
            f' - Best train RMSE score: {round(gs.best_score_, 3)}'
        )

        # log results
        regr_result_log['best_estimator'] = {}
        regr_result_log['best_estimator'] = {
            'estimator': estimator,
            'neg_rmse_train': gs.best_score_,
            'neg_rmse_test': gs.score(X_test, y_test),
            'r2_train': r2_train,
            'r2_test': r2_test,
            'best_params': gs.best_params_
        }

        return regr_result_log

def train_estimators(regr_results_log, X_train, y_train):
    """
    Train default estimators using cross_val_score and return dictonary with
    estimators and their scores.

    Parameters
    ----------
    regr_results_log : dict{str: dict{str: any}}
        Dictionary with regression propertires for current regression
    X_train : numpy.ndarray
        Training data regressor
    y_train : numpy.ndarray
        Training data regressand
    
    Returns
    -------
    regr_results_log : dict{str: dict{str: any}}
        Dictionary with regression propertires for current regression including
        results of estimators
    """
    # initialize estimators
    estimators = [
        BayesianRidge(),
        DummyRegressor(),
        ElasticNet(),
        GradientBoostingRegressor(),
        Lasso(),
        LinearRegression(),
        RandomForestRegressor(),
        Ridge(),
        SGDRegressor(),
        SVR(),
    ]

    regr_results_log['regr_results'] = {}
    # iterate through estimators
    for estimator in estimators:
        cv_results = cross_val_score(
            estimator,
            X_train, y_train,
            cv=20,
            scoring='neg_root_mean_squared_error',
            # verbose=1
        )

        # log results
        regr_results_log['regr_results'][estimator.__class__.__name__] = {
            'neg_rmse': cv_results.mean(),
            'neg_rmse_std': cv_results.std()
        }
        
        print(f'\t{cv_results.mean():.3f} ({cv_results.std():.3f}) : {estimator.__class__.__name__}')

    return regr_results_log


def filter_outliers(
    df_regression, regressor, use_grouping=False, zscore_threshold=0.0
    ):
        """
        Filter outlier based on regressor metric and z-score threshold.
        If z-score threshold is 0.0, no outlier filtering is performed.
        If data is grouped, no outlier filtering is performed.

        Parameters
        ----------
        df_regression : pandas.DataFrame
            Pandas dataframe with regression report results for current regression
        regressor : str
            Regressor metric to be filtered for
        use_grouping : bool
            if data is grouped by any variable
        zscore_threshold : float
            Threshold of z-score for outlier filtering based on regressor

        Returns
        -------
        df_regression : pandas.DataFrame
            Pandas dataframe with regression report results for current regression
            filtered by z-score threshold
        """
        # filter outlier of regressor (if data is not aggregated)
        n_filtered = 0
        if zscore_threshold and not use_grouping:
            # create mask based on zscore threshold
            mask = np.abs(stats.zscore(df_regression[regressor])) > zscore_threshold
            n_filtered = mask.sum()

            # apply mask
            df_regression[regressor] = df_regression[regressor].mask(mask)

            # drop outliers
            df_regression.dropna(inplace=True)
        
        print(f'\t{n_filtered} outliers filtered')
        return df_regression

def filter_data(
    use_grouping, group_by, df_regression,
    cnn=False, classes=False
    ):
        """
        Filter data for specific model, classification task and grouping.
        
        Parameters
        ----------
        use_grouping : bool
            Use grouping
        group_by : dict{str: bool}
            Dictonary with vars to group data by
        df_regression : pandas.DataFrame
            Pandas dataframe with regression report results for current regression
        cnn : str
            Model to filter for ('resnet', 'basic'), False for all (default)
        classes : int
            Classification task to be filtered for, False for all (default)
        
        Returns
        -------
        df_regression : pandas.DataFrame
            Pandas dataframe with regression report results filtered by current
            regression parameter
        """
        if cnn:
            df_regression = df_regression.query(f'model == "{cnn}"')
        if classes:
            df_regression = df_regression.query(f'classes == "{classes}"')
        if use_grouping:
            df_regression = df_regression \
                .groupby([k for k, v in group_by.items() if v]) \
                .mean().reset_index()
        
        return df_regression

def calc_delta(df_raw, df_regression, regressor='accuracy'):
    """
    Calculate delta between regressand of each experiment setup and average of
    resepective setup with 0.0% false labels ratio.
        -> model + class + metric

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Pandas dataframe with raw classification report results
    df_regression : pandas.DataFrame
        Pandas dataframe with regression report results for current regression

    regressor : str
        Regressor metric to be filtered for, default 'accuracy'
    
    Returns
    -------
    df_regression : pandas.DataFrame
        Pandas dataframe with regression report results for current regression
        including delta column
    """
    # calc delta to baseline of average 0 ratio + model + class setup
    df_tmp = df_raw.groupby(['model', 'classes', 'ratio', 'metric']) \
        .mean().reset_index()
    df_regression[f'{regressor}_delta'] = df_regression.apply(lambda row:
        row[regressor] -
        df_tmp.query(
            f'model=="{row.model}" & classes=="{row.classes}" & ratio==0.0 & metric=="{row.metric}"'
        ).iloc[0]['value'], axis=1
    )

    return df_regression

def regr_results_logger(
    group_by,
    regressor='accuracy',
    cnn=False,
    classes=False,
    use_delta=False,
    zscore_threshold=0.0
    ):
        """
        Initialize dict with regression properties to be logged.

        Parameters
        ----------
        group_by : dict{str: bool}
            Dictonary with vars to group data by
        regressor : str
            Regressor metric to be filtered for, default 'accuracy'
        cnn : str
            Model to filter for ('resnet', 'basic'), False for all (default)
        classes : int
            Classification task to be filtered for, False for all (default)
        use_delta : bool
            Use regressor delta to 0 ratio experiment setup for regression,
            default False
        zscore_threshold : float
            Threshold of z-score for regressand outlier filtering,
            0.0 for no outlier filtering (default)

        Returns
        -------
        regr_props : dict
            Dictonary with regression properties to be logged
        """
        regr_props = {}
        regr_props['properties'] = {}
        regr_props['properties']['type'] = 'regression'
        regr_props['properties']['regressor'] = regressor
        regr_props['properties']['regressand'] = 'false_labels_ratio'
        regr_props['properties']['model'] = cnn if cnn else 'all'
        regr_props['properties']['classification_task'] = classes if classes else 'all'
        regr_props['properties']['delta'] = use_delta
        regr_props['properties']['group_by'] = group_by
        regr_props['properties']['zscore_threshold'] = zscore_threshold

        return regr_props

def load_aggr_class_reports(aggr_classReport_path):
    """
    Aggregate classification reports from multiple model training runs filtered
    for a specific epoch. Saves the aggregated report to a csv file.

    Parameters
    ----------
    aggr_classReport_path : str
        Path to the aggregated classification report csv

    Returns
    -------
    df : pandas.DataFrame
        Pandas dataframe with aggregated classification report results
    """
    # load aggregated classification reports
    df = pd.read_csv(aggr_classReport_path + 'class_report_metrics.csv')
    df['classes'] = df['classes'].astype('str')
    print(f'\t{int(df.shape[0] / len(df["metric"].unique()))} classification reports loaded - df.shape {df.shape}')

    return df

def aggr_class_reports(report_log_path, filter, csv_path):
    """
    Aggregate classification reports from multiple model training runs filtered
    for a specific epoch. Saves the aggregated report to a csv file.

    Parameters
    ----------
    report_log_path : str or path-like
        Path to the classification report log directory
    filter : str
        File name filter for the classification report log files,
        e.g. to filter only epoch 19 class. reports: 'class_report_19epoch.json'
    csv_path : str or path-like
        Path to where to save the csv file with aggregated reports to

    Returns
    -------
    df : pandas.DataFrame
        Pandas dataframe with aggregated classification report results
        as per filter
    """
    # define metrics to be extracted as list
    logs = [['accuracy']
            , ['macro avg', 'precision']
            , ['macro avg', 'recall']
            , ['macro avg', 'f1-score']
            , ['weighted avg', 'precision']
            , ['weighted avg', 'recall']
            , ['weighted avg', 'f1-score']
    ]

    # init dataframe
    df_headers = ['run', 'model', 'ratio', 'classes', 'metric', 'value']
    df = pd.DataFrame(columns=df_headers)

    # iter through logs and get metrics from classification report jsons
    counter = 0
    for (dirpath, dirnames, filenames) in os.walk(report_log_path):

        if filter in filenames:
            counter += 1

            # load classification report
            file_path = dirpath + '/' + filter
            class_report = json.load(open(file_path))

            for log in logs:
                # extract meta data from file name
                run = dirpath.split('\\', 1)[-1].replace('\\', '_')
                model = re.search('(^[a-z]{5,6})_', run).group(1)
                ratio = float(re.search('(\d{5})r', run).group(1)) / 100
                classes = re.search('(\d{1,2})c', run).group(1)
                
                if len(log) == 1:
                    metric = log[0]
                    value = class_report[log[0]]
                elif len(log) == 2:
                    metric = log[0] + ' ' + log[1]
                    value = class_report[log[0]][log[1]]
                
                # append to dataframe
                row = [
                    str(run),
                    str(model),
                    float(ratio),
                    str(classes),
                    str(metric),
                    float(value)
                ]
                df_len = len(df)
                df.loc[df_len] = row

    print(f'\t{counter} classification reports aggregated - df.shape {df.shape}')
    df.to_csv(csv_path + 'class_report_metrics.csv', index=False)

    return df

def init_estimator(estimator_name):
    """
    Initialize estimator and grid search params based on estimator name.

    Parameters
    ----------
    estimator_name : str
        Class name of estimator to be tuned

    Returns
    -------
    estimator : sklearn estimator class
        Estimator to be tuned
    params : dict{str: any}
        Grid search params for estimator
    """
    # initialize estimators and hyperparameter params for grid search
    
    if estimator_name == 'BayesianRidge':
        estimator = BayesianRidge()
        params = {}
        # params['max_iter'] = [10000]
        params['alpha_1'] = [1e-6, 1e-5, 1e-4]
        params['alpha_2'] = [1e-6, 1e-5, 1e-4]
        params['lambda_1'] = [1e-6, 1e-5, 1e-4]
        params['lambda_2'] = [1e-6, 1e-5, 1e-4]
    
    elif estimator_name == 'DummyRegressor':
        estimator = DummyRegressor()
        params = {}
        params['strategy'] = ['mean', 'median', 'quantile']
        params['quantile'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    elif estimator_name == 'ElasicNet':
        estimator = ElasticNet()
        params = {}
        params['max_iter'] = [10000]
        params['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        params['l1_ratio'] = np.linspace(0.001, 0.999, 10)
        params['selection'] = ['random', 'cyclic']

    elif estimator_name == 'GradientBoostingRegressor':
        estimator = GradientBoostingRegressor()
        params = {}
        params['loss'] = ['squared_error', 'huber']
        params['learning_rate'] = [0.1, 0.2, 0.5, 0.7]
        params['n_estimators'] = [100, 200, 400]
        # params['criterion'] = ['friedman_mse', 'squared_error', 'absolute_error']
        params['max_depth'] = [1, 2, 4]
        params['alpha'] = [0.1, 0.5, 0.9]

    elif estimator_name == 'Lasso':
        estimator = Lasso()
        params = {}
        params['max_iter'] = [10000]
        params['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        params['selection'] = ['random', 'cyclic']

    elif estimator_name == 'LinearRegression':
        estimator = LinearRegression()
        params = {}

    elif estimator_name == 'RandomForestRegressor':
        estimator = RandomForestRegressor()
        params = {}
        params['n_estimators'] = [100, 200, 300, 400, 500]
        params['criterion'] = ['squared_error'] # 'absolute_error', 'poisson'
        params['max_depth'] = [1, 2, 3, 4, 5]

    elif estimator_name == 'Ridge':
        estimator = Ridge()
        params = {}
        params['max_iter'] = [1000]
        params['alpha'] = [1e-2, 1e-1, 1.0, 10.0, 100.0]

    elif estimator_name == 'SGDRegressor':
        estimator = SGDRegressor()
        params = {}
        params['loss'] = ['squared_error', 'huber', 'epsilon_insensitive']
        params['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        params['max_iter'] = [10000]
        params['epsilon'] = [1e-2, 1e-1, 1.0]
        params['learning_rate'] = ['constant', 'optimal', 'invscaling', 'adaptive']

    elif estimator_name == 'SVR':
        estimator = SVR()
        params = {}
        params['kernel'] = ['linear', 'poly'] # 'rbf', 'sigmoid'
        params['degree'] = [2, 3, 4]
        params['gamma'] = ['scale', 'auto']
        params['coef0'] = [0.0, 0.1, 0.3, 0.8]
        params['C'] = [1e-1, 1.0, 10.0]
        # params['epsilon'] = [1e-2, 1e-1, 1.0]

    else:
        print('Estimator not found')
        exit()

    return estimator , params
