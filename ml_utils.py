"""
ML Utils for regression problems from WW input
"""
import datetime
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import sklearn

from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from matplotlib.axis import Axis

import statsmodels
import statsmodels.api as sm


class Dataset():
    """Convenient class to prepare WW dataset for regression
    Performs  variable selection, input/output transformation, filtering, train/test splitting and standardisation
    """
    def __init__(self, 
                 data,
                 input_variables, 
                 target_variable,
                 y_scale = 'log10',
                 log10_variables=['sars_cov2_gc_l_mean', 'control_gc_l_mean',  'suspended_solids_mg_l',
                                  'ammonia_mg_l', 'ophosph_mg_l'],
                 date_column='date',
                 sars_above=None,
                 sars_below=None,
                 target_above=None,
                 min_date=None,
                 max_date=None,
                 input_offset = 0.,
                 target_offset = 0.
                 ):
        """Create an instance of Dataset class
        
        Args:
            data: pandas DataFrame
                contain input_variables and target_variables
            input_variables: list
                list of columns to consider
            target_variable: str
                target name
            y_scale: str
                one of log, log10 or linear
            log10_variables:
                input variables to log
            date_column: str:
                which date column
            sars_above: float
                filter for minimal sars value
            sars_below: float
                filter for max sars value
            target_above: float
                target name
            max_date: date
                filter for max date
            input_offset: float
                constant to add to the input before transformation
            target_offset: float
                constant to add to the target before transfro
        """
        
        self.input_variables = input_variables
        self.target_variable = target_variable
        self.log10_variables = log10_variables
        self.date_column = date_column
        self.y_scale = y_scale
        
        dsata = data.copy()
        if sars_above is not None:
            data = data[data.sars_cov2_gc_l_mean>sars_above]
        if sars_below is not None:
            data = data[data.sars_cov2_gc_l_mean<sars_below]
        if target_above is not None:
            data = data[data[target_variable]>target_above]
        if max_date is not None:
            names = data.index.names
            data = data.reset_index()[data.reset_index()['date']<=max_date].set_index(names)
        if min_date is not None:
            names = data.index.names
            data = data.reset_index()[data.reset_index()['date']>=min_date].set_index(names)
        
        self.X = data[input_variables]
        self.Y = data[target_variable]

        self.scaler = StandardScaler()
        
        
        if self.y_scale == 'linear':
            self.y_transform = lambda x: x + target_offset
            self.inverse_transform_y = lambda x: x - target_offset
        elif self.y_scale == 'log':
            self.y_transform = lambda x: np.log(x + target_offset)
            self.inverse_transform_y = lambda x: np.exp(x) - target_offset
        elif self.y_scale == 'log10':
            self.y_transform = lambda x: np.log10(x + target_offset)
            self.inverse_transform_y = lambda x: 10**x - target_offset
        else:
            raise ValueError('y_scale must be linear or log')
            
        self.inverse_transform_x = dict()
        self.input_offset = input_offset
        self.x_transform = dict()
        
    def log10_transform(self):
        x = self.X.copy()
        # Log transfo
        
        for var in self.input_variables:
            if var in self.log10_variables:
                x[var] = np.log10(x[var]+self.input_offset)
                self.x_transform[var] = lambda x: np.log10(x + self.input_offset)
                self.inverse_transform_x[var] = lambda y: 10**y - self.input_offset
            else:
                self.x_transform[var] = lambda x: x
                self.inverse_transform_x[var] = lambda y: y

        y = self.y_transform(self.Y)
        return x, y
    
        
    def random_split(self, test_size = 0.25, random_state = 0, standardise=False):
        x, y = self.log10_transform()
        
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, 
                                                                                    test_size=test_size, 
                                                                                    shuffle=True, 
                                                                                    random_state=random_state)
        if standardise:
            x_train.loc[:,:] = self.scaler.fit_transform(x_train)
            x_test.loc[:,:] = self.scaler.transform(x_test)
        self.x = x_train.append(x_test)
        self.y = y_train.append(y_test)
        return x_train, x_test, y_train, y_test
        
    def temporal_split(self, start_date_test=None, recent_days=None, standardise=False):
        x, y = self.log10_transform()
        
        index = x.index.names
        x = x.reset_index()
        y = y.reset_index()

        if recent_days is not None:
            start_date_test = x[self.date_column].max() - datetime.timedelta(recent_days)
        
        x_train = x[x[self.date_column] < start_date_test].set_index(index)
        y_train = y[y[self.date_column] < start_date_test].set_index(index)[self.target_variable]
        x_test = x[x[self.date_column] >= start_date_test].set_index(index)
        y_test = y[y[self.date_column] >= start_date_test].set_index(index)[self.target_variable]
        
            
        if standardise:
            x_train.loc[:,:] = self.scaler.fit_transform(x_train)
            x_test.loc[:,:] = self.scaler.transform(x_test)
        self.x = x_train.append(x_test)
        self.y = y_train.append(y_test)
        return x_train, x_test, y_train, y_test
        
    def prepare_no_split(self, standardise=False):
        x, y = self.log10_transform()
        if standardise:
            x.loc[:,:] = self.scaler.fit_transform(x)
        self.x = x
        self.y = y 
        return x, y    
            

def predict_kfold(model, x, y, n_splits=5, shuffle=True, random_state=0):
    """ Return a prediction vector for a sklearn-type model using a K-Fold procedure
    
    Args
        x: dataframe 
            inputs
        y: dataframe 
            outputs
        n_splits: int
            number of splits for KFold (5 by default)
        shuffle: bool
            KFold arg (True by default)
        randome_state: int
            KFold arg (0 by default)
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    pred = y.copy()
    pred[:] = np.nan
    
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, _ = y.iloc[train_index], y.iloc[test_index]
        model = sklearn.base.clone(model)
        model.fit(x_train, y_train)
        pred.iloc[test_index] = model.predict(x_test)
    return pred


def bootstrap(model, x, y, x_eval = None, repeat=1, test_size=0.2, shuffle=True, random_seed=None):
    """Return a dataframe of bootstrapped predictions for a sklearn-type model using a number of random train/test splits
    
    Args
        model:
            sklearn-type model (clonable with fit method)
        x: dataframe 
            inputs
        y: dataframe 
            outpus
        x_eval: dataframe
            out-of sample intputs to produce predictions for
        repeat: int
            number of splits for KFold
        test_size: float
            fractional test size (between 0 and 1)
        shuffle: whether to shuffle the dataset
        random_seed: int
            for numpy init (default = None)
    """
    pred = pd.DataFrame(y.copy()).drop(y.name, axis=1)
    if x_eval is not None:
        pred_eval = pd.DataFrame(index=x_eval.index)
    
    np.random.seed(random_seed)
    for i in tqdm.tqdm(range(repeat)):

        train_size = 0.8
        train_index = np.random.choice(range(len(x)), size=int((1-test_size)*len(x)), replace=False)
        test_index = np.array([n for n in range(len(x)) if n not in train_index])
        
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, _ = y.iloc[train_index], y.iloc[test_index]
        model = sklearn.base.clone(model)
        model.fit(x_train, y_train)
        pred[f'boot_{i}'] = np.nan
        pred[f'boot_{i}'].iloc[test_index] = model.predict(x_test)
        if x_eval is not None:
            pred_eval[f'boot_{i}'] = model.predict(x_eval)
    if x_eval is not None:
        return pred, pred_eval
    else:
        return pred


class RandomIntercepts(BaseEstimator, RegressorMixin):
    """ A sklearn-style wrapper statsmodels MixedLM model for random intercepts """
    def __init__(self, groupvar):
        self.groupvar = groupvar
        
    def fit(self, x, y):
        self.model_ = sm.MixedLM(y, x, groups=x.reset_index()[self.groupvar], missing='ignore' )
        self.results_ = self.model_.fit()
        
    def predict(self, x):
        intercepts = x.reset_index().reset_index().set_index(self.groupvar).join(pd.DataFrame(self.results_.random_effects).T).sort_values('index')['Group Var']
        return self.results_.predict(exog=x) + intercepts.values

    @property
    def coef_(self):
        return [value for var, value in self.results_.fe_params.items()]
    
    
class RandomEffects(BaseEstimator, RegressorMixin):
    """ A sklearn-style wrapper statsmodels MixedLM model for random slopes and intercepts """
    def __init__(self, groupvar, slope_vars=None, correlated_re=False):
        self.groupvar = groupvar
        self.slope_vars = slope_vars
        self.correlated_re = correlated_re
        
    def fit(self, x, y):
        data = x.join(y)
        target = data.columns[-1]
        
        if self.slope_vars is None:
            self.slope_vars = list(x.columns)
        else:
            assert len([var for var in self.slope_vars if var not in x.columns]) == 0
            
        if not self.correlated_re:
            free = statsmodels.regression.mixed_linear_model.MixedLMParams.from_components(
                                         fe_params=np.ones(data.shape[1]),
                                         cov_re=np.eye(1+len(self.slope_vars)))
        else:
            free = None
            
        self.model_ = sm.MixedLM.from_formula(f"{target} ~ " + ' + '.join([f"{var}" for var in x.columns]), 
                                                data, 
                                                re_formula=' + '.join([f"{var}" for var in self.slope_vars]), # by default intercept  and slope
                                                groups=data.reset_index()[self.groupvar].values)
        self.results_ = self.model_.fit(free=free, method=["lbfgs"])
        
    def predict(self, x):
        # fixed effect
        fe = self.results_.predict(x)
        # Random effect
        df_re = pd.DataFrame(self.results_.random_effects).T
        group_intercepts = x.reset_index().reset_index().set_index(self.groupvar)[['index']].join(df_re[['Group']]).sort_values('index')['Group']
        group_slopes = x.reset_index().reset_index().set_index(self.groupvar)[['index']].join(df_re[self.slope_vars]).sort_values('index')[self.slope_vars]
        re = (x[self.slope_vars] * group_slopes.values).sum(1) + group_intercepts.values.flatten()
        return fe + re

    @property
    def coef_(self):
        return [value for var, value in self.results_.fe_params.items() if var!= 'Intercept']
    

def plot_conditional_pred(y, pred, bins=(10, 20), ax=None, title=None):
    """Plot colour-map of predictions as a function of True values
    
    Args:
        y: array-like
            True values
        pred: array-like
            predictions (same length as y)
        bins: 2-tuple/2-list
            binning values for x (true values) and y (preds) axes
        ax: matplotlib axe
            optional axis to populate 
        title:
            optional title
    Return:
        corresponding plt.axe object"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = Axis.get_figure(ax)
    ranges = [[np.min(y), np.max(y)]]*2
    h,  bins_y, bins_pred = np.histogram2d(y, pred, bins=bins, range=ranges)
    h_norm = (h / h.sum(1, keepdims=True)).T

    ax.scatter(y, pred, s=0.2, alpha=0.1, color='red')
    ax.plot(*ranges, 
             c='red', label='$\hat{y}=y$', alpha=0.5)
    im = ax.imshow(h_norm, 
               extent=np.array(ranges).flatten(),  
               cmap=plt.cm.Reds, origin='lower', interpolation='nearest')

    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel(f'Proportion of estimates', fontsize=14, rotation=270, labelpad=20)
    
    ax.set_xlabel('True values', fontsize=14)
    ax.set_ylabel(f'Predictions', fontsize=14)
    ax.set_title(title)
    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    return ax
