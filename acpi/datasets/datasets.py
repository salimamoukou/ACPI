import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.stats as st
import os
from . import config

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


base_path = "https://github.com/salimamoukou/ACPI/tree/main/datasets/"

class simulation:
    """ Functions for generating 1 dimensional simulation
    Parameters
    ----------
    rank: int, the number of simulation function
    """
    def __init__(self, rank) -> None:
        self.rank = rank

    def f_1(self, x):
        return np.sin(x) ** 2 + 0.1 + 0.6 * np.sin(2 * x) * np.random.randn(1)

    def f_2(self, x, gt=False):
        if not gt:
            return 2 * np.sin(x) ** 2 + 0.1 + 0.15 * x * np.random.randn(1)
        else:
            noise = np.random.randn(1)
            data = 2 * np.sin(x) ** 2 + 0.1
            return data, noise

    def f_3(self, x):
        x = np.random.poisson(np.sin(x) ** 2 + 0.1) + 0.08 * x * np.random.randn(1)
        x += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
        return x
    
    def f(self, x):
        x = np.random.poisson(np.sin(x) ** 2 + 0.1) + 0.03 * x * np.random.randn(1)
        x += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
        return x

    def generate(self, data, gt=False):
        y = 0 * data
        noise = 0 * data
        for i in range(len(data)):
            if self.rank == 1:
                y[i] = self.f_1(data[i])
            elif self.rank == 2:
                if gt:
                    y[i], noise[i] = self.f_2(data[i], gt=gt)
                else:
                    y[i] = self.f_2(data[i])
            elif self.rank == 3:
                y[i] = self.f_3(data[i])
            else:
                y[i] = self.f(data[i])
        if gt:
            return y.astype(np.float32), noise.astype(np.float32)
        else:
            return y.astype(np.float32)


class GaussianDataGenerator(object):
    def __init__(self, px_model, mu_model, sigma_model):
        self.px_model = px_model
        self.mu_model = mu_model
        self.sigma_model = sigma_model
    
    def generate(self, size, **kwargs):
        if 'a' in kwargs:
            a = kwargs.pop('a')
            b = kwargs.pop('b')
            X = self.px_model(size, a=a, b=b)
        else:
            X = self.px_model(size)
        Y = self.mu_model(X) + self.sigma_model(X) * np.random.randn(size)
        return X, Y


def px_model(size, **kwargs):
    if 'a' in kwargs:
        a = kwargs.pop('a')
        b = kwargs.pop('b')
        return config.DataParams.left_interval + (
                    config.DataParams.right_interval - config.DataParams.left_interval) * np.expand_dims(beta.rvs(a, b, size=size), 1)
    return config.DataParams.left_interval + (config.DataParams.right_interval - config.DataParams.left_interval) * np.random.rand(size, 1)


def mu_model(x):
    k = [1.0]
    b = -0.0
    return np.sum(k * x, axis=-1) + b 


def sigma_model(x):
    x_abs = np.abs(x)
    return (x_abs / (x_abs + 1)).reshape(-1)


def GetDataset(name, seed, test_ratio, a=1., b=1.):

    if 'simulation' in name:
        x_train = np.random.uniform(0, 5.0, size=config.DataParams.n_train).astype(np.float32)
        x_test = np.random.uniform(0, 5.0, size=config.DataParams.n_test).astype(np.float32)

        if '1' in name:
            sim = simulation(1)
        elif '2' in name:
            sim = simulation(2)
        elif '3' in name:
            sim = simulation(3)
        else:
            sim = simulation(4)

        y_train = sim.generate(x_train)
        y_test = sim.generate(x_test)
        x_train = np.reshape(x_train, (config.DataParams.n_train, 1))
        x_test = np.reshape(x_test, (config.DataParams.n_test, 1))
    
    if name == 'cov_shift':
        data_model = GaussianDataGenerator(px_model, mu_model, sigma_model)
        x_train, y_train = data_model.generate(config.DataParams.n_train)
        x_test, y_test = data_model.generate(config.DataParams.n_test, a=a, b=b)

    if name=="meps_19":
        df = pd.read_csv(cache(base_path + 'meps_19_reg.csv'))
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names!=response_name]
        column_names = column_names[column_names!="Unnamed: 0"]
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
        
    if name=="meps_20":
        df = pd.read_csv(cache(base_path + 'meps_20_reg.csv'))
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names!=response_name]
        column_names = column_names[column_names!="Unnamed: 0"]
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    if name=="meps_21":
        df = pd.read_csv(cache(base_path + 'meps_21_reg.csv'))
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names!=response_name]
        column_names = column_names[column_names!="Unnamed: 0"]
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    if name=="star":
        df = pd.read_csv(cache(base_path + 'STAR.csv'))
        df.loc[df['gender'] == 'female', 'gender'] = 0
        df.loc[df['gender'] == 'male', 'gender'] = 1
        
        df.loc[df['ethnicity'] == 'cauc', 'ethnicity'] = 0
        df.loc[df['ethnicity'] == 'afam', 'ethnicity'] = 1
        df.loc[df['ethnicity'] == 'asian', 'ethnicity'] = 2
        df.loc[df['ethnicity'] == 'hispanic', 'ethnicity'] = 3
        df.loc[df['ethnicity'] == 'amindian', 'ethnicity'] = 4
        df.loc[df['ethnicity'] == 'other', 'ethnicity'] = 5
        
        df.loc[df['stark'] == 'regular', 'stark'] = 0
        df.loc[df['stark'] == 'small', 'stark'] = 1
        df.loc[df['stark'] == 'regular+aide', 'stark'] = 2
        
        df.loc[df['star1'] == 'regular', 'star1'] = 0
        df.loc[df['star1'] == 'small', 'star1'] = 1
        df.loc[df['star1'] == 'regular+aide', 'star1'] = 2        
        
        df.loc[df['star2'] == 'regular', 'star2'] = 0
        df.loc[df['star2'] == 'small', 'star2'] = 1
        df.loc[df['star2'] == 'regular+aide', 'star2'] = 2   

        df.loc[df['star3'] == 'regular', 'star3'] = 0
        df.loc[df['star3'] == 'small', 'star3'] = 1
        df.loc[df['star3'] == 'regular+aide', 'star3'] = 2      
        
        df.loc[df['lunchk'] == 'free', 'lunchk'] = 0
        df.loc[df['lunchk'] == 'non-free', 'lunchk'] = 1
        
        df.loc[df['lunch1'] == 'free', 'lunch1'] = 0    
        df.loc[df['lunch1'] == 'non-free', 'lunch1'] = 1      
        
        df.loc[df['lunch2'] == 'free', 'lunch2'] = 0    
        df.loc[df['lunch2'] == 'non-free', 'lunch2'] = 1  
        
        df.loc[df['lunch3'] == 'free', 'lunch3'] = 0    
        df.loc[df['lunch3'] == 'non-free', 'lunch3'] = 1  
        
        df.loc[df['schoolk'] == 'inner-city', 'schoolk'] = 0
        df.loc[df['schoolk'] == 'suburban', 'schoolk'] = 1
        df.loc[df['schoolk'] == 'rural', 'schoolk'] = 2  
        df.loc[df['schoolk'] == 'urban', 'schoolk'] = 3

        df.loc[df['school1'] == 'inner-city', 'school1'] = 0
        df.loc[df['school1'] == 'suburban', 'school1'] = 1
        df.loc[df['school1'] == 'rural', 'school1'] = 2  
        df.loc[df['school1'] == 'urban', 'school1'] = 3      
        
        df.loc[df['school2'] == 'inner-city', 'school2'] = 0
        df.loc[df['school2'] == 'suburban', 'school2'] = 1
        df.loc[df['school2'] == 'rural', 'school2'] = 2  
        df.loc[df['school2'] == 'urban', 'school2'] = 3      
        
        df.loc[df['school3'] == 'inner-city', 'school3'] = 0
        df.loc[df['school3'] == 'suburban', 'school3'] = 1
        df.loc[df['school3'] == 'rural', 'school3'] = 2  
        df.loc[df['school3'] == 'urban', 'school3'] = 3  
        
        df.loc[df['degreek'] == 'bachelor', 'degreek'] = 0
        df.loc[df['degreek'] == 'master', 'degreek'] = 1
        df.loc[df['degreek'] == 'specialist', 'degreek'] = 2  
        df.loc[df['degreek'] == 'master+', 'degreek'] = 3 

        df.loc[df['degree1'] == 'bachelor', 'degree1'] = 0
        df.loc[df['degree1'] == 'master', 'degree1'] = 1
        df.loc[df['degree1'] == 'specialist', 'degree1'] = 2  
        df.loc[df['degree1'] == 'phd', 'degree1'] = 3              
        
        df.loc[df['degree2'] == 'bachelor', 'degree2'] = 0
        df.loc[df['degree2'] == 'master', 'degree2'] = 1
        df.loc[df['degree2'] == 'specialist', 'degree2'] = 2  
        df.loc[df['degree2'] == 'phd', 'degree2'] = 3
        
        df.loc[df['degree3'] == 'bachelor', 'degree3'] = 0
        df.loc[df['degree3'] == 'master', 'degree3'] = 1
        df.loc[df['degree3'] == 'specialist', 'degree3'] = 2  
        df.loc[df['degree3'] == 'phd', 'degree3'] = 3          
        
        df.loc[df['ladderk'] == 'level1', 'ladderk'] = 0
        df.loc[df['ladderk'] == 'level2', 'ladderk'] = 1
        df.loc[df['ladderk'] == 'level3', 'ladderk'] = 2  
        df.loc[df['ladderk'] == 'apprentice', 'ladderk'] = 3  
        df.loc[df['ladderk'] == 'probation', 'ladderk'] = 4
        df.loc[df['ladderk'] == 'pending', 'ladderk'] = 5
        df.loc[df['ladderk'] == 'notladder', 'ladderk'] = 6
        
        
        df.loc[df['ladder1'] == 'level1', 'ladder1'] = 0
        df.loc[df['ladder1'] == 'level2', 'ladder1'] = 1
        df.loc[df['ladder1'] == 'level3', 'ladder1'] = 2  
        df.loc[df['ladder1'] == 'apprentice', 'ladder1'] = 3  
        df.loc[df['ladder1'] == 'probation', 'ladder1'] = 4
        df.loc[df['ladder1'] == 'noladder', 'ladder1'] = 5
        df.loc[df['ladder1'] == 'notladder', 'ladder1'] = 6
        
        df.loc[df['ladder2'] == 'level1', 'ladder2'] = 0
        df.loc[df['ladder2'] == 'level2', 'ladder2'] = 1
        df.loc[df['ladder2'] == 'level3', 'ladder2'] = 2  
        df.loc[df['ladder2'] == 'apprentice', 'ladder2'] = 3  
        df.loc[df['ladder2'] == 'probation', 'ladder2'] = 4
        df.loc[df['ladder2'] == 'noladder', 'ladder2'] = 5
        df.loc[df['ladder2'] == 'notladder', 'ladder2'] = 6
        
        df.loc[df['ladder3'] == 'level1', 'ladder3'] = 0
        df.loc[df['ladder3'] == 'level2', 'ladder3'] = 1
        df.loc[df['ladder3'] == 'level3', 'ladder3'] = 2  
        df.loc[df['ladder3'] == 'apprentice', 'ladder3'] = 3  
        df.loc[df['ladder3'] == 'probation', 'ladder3'] = 4
        df.loc[df['ladder3'] == 'noladder', 'ladder3'] = 5
        df.loc[df['ladder3'] == 'notladder', 'ladder3'] = 6
        
        df.loc[df['tethnicityk'] == 'cauc', 'tethnicityk'] = 0
        df.loc[df['tethnicityk'] == 'afam', 'tethnicityk'] = 1
        
        df.loc[df['tethnicity1'] == 'cauc', 'tethnicity1'] = 0
        df.loc[df['tethnicity1'] == 'afam', 'tethnicity1'] = 1
        
        df.loc[df['tethnicity2'] == 'cauc', 'tethnicity2'] = 0
        df.loc[df['tethnicity2'] == 'afam', 'tethnicity2'] = 1
        
        df.loc[df['tethnicity3'] == 'cauc', 'tethnicity3'] = 0
        df.loc[df['tethnicity3'] == 'afam', 'tethnicity3'] = 1
        df.loc[df['tethnicity3'] == 'asian', 'tethnicity3'] = 2
        
        df = df.dropna()
        
        grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
        grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]
        
        
        names = df.columns
        target_names = names[8:16]
        data_names = np.concatenate((names[0:8],names[17:]))
        X = df.loc[:, data_names].values
        y = grade.values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    if name=="bio":
        #https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(cache(base_path + 'CASP.csv'))
        df.fillna(0, inplace=True) 
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    if name=="bike":
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df=pd.read_csv(cache(base_path + 'bike_train.csv'))
        
        # # seperating season as per values. this is bcoz this will enhance features.
        season=pd.get_dummies(df['season'],prefix='season')
        df=pd.concat([df,season],axis=1)
        
        # # # same for weather. this is bcoz this will enhance features.
        weather=pd.get_dummies(df['weather'],prefix='weather')
        df=pd.concat([df,weather],axis=1)
        
        # # # now can drop weather and season.
        df.drop(['season','weather'],inplace=True,axis=1)
        df.head()
        
        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011:0, 2012:1})
 
        df.drop('datetime',axis=1,inplace=True)
        df.drop(['casual','registered'],axis=1,inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        X = df.drop('count',axis=1).values
        y = df['count'].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    if name == "concrete":
        dataset = np.loadtxt(open(cache(base_path + 'Concrete_Data.csv', "rb")), delimiter=",", skiprows=1)
        X = dataset[:, :-1]
        y = dataset[:, -1:]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    if name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(cache(base_path + 'communities_attributes.csv', delim_whitespace = True))
        data = pd.read_csv(cache(base_path + 'communities.data', names = attrib['attributes']))
        data = data.drop(columns=['state','county', 'community','communityname', 'fold'], axis=1)
        data = data.replace('?', np.nan)
        
        # Impute mean values for samples with missing values        
        from sklearn.impute import SimpleImputer
        
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        
        imputer = imputer.fit(data[['OtherPerCap']])
        data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    if name == 'generate_1_dim':
        n = config.DataParams.n_train
        d = config.DataParams.d
        eps = st.norm(0, 1e-10)
        x_law = st.uniform(0, 7)
        X = x_law.rvs(n * d).reshape(n, d)
        y = generate_y(X, gen_f1, eps)
        x_train, x_test, y_train, y_test = train_test_split(X.astype(np.double), y.astype(np.double),
                                                            test_size=config.DataParams.test_ratio,
                                                            random_state=config.UtilsParams.seed)

    if name == 'generate_3_dim':
        eps_x = st.norm(0, 0.5)
        def gen_f3(X):
            X = 2 * (X - 0.5)
            y = X[0] * X[1] + (2 * X[1] - 1) ** 2 + np.sin(2 * np.pi * X[2]) / (2 - np.sin(2 * np.pi * X[2])) + eps_x.rvs(
                1) * np.sin(2 * np.pi * X[4])
            return y
        n = 5000
        d = 50
        eps = st.norm(0, 1e-5)
        x_law = st.uniform(0, 1)
        X = x_law.rvs(n * d).reshape(n, d)
        y = generate_y(X, gen_f3, eps)
        x_train, x_test, y_train, y_test = train_test_split(X.astype(np.double), y.astype(np.double),
                                                            test_size=config.DataParams.test_ratio,
                                                            random_state=config.UtilsParams.seed)
    if name == 'generate_9_dim':
        # eps_x = st.norm(0, 0.5)
        def gen_f4(X):
            fun_in = X[0] + (X[0]/(1 + X[0])) * np.random.randint(1)
            # y = (X[1]>0) + X[2]**3 + (X[4] + X[6] - X[8] - X[9] > 1 + X[10]) + np.exp(-X[2]**2) + eps_x.rvs(1)*np.sin(2*np.pi*X[11])
            return fun_in
        n = 5000
        d = 50
        eps = st.norm(0, 1e-5)
        x_law = st.uniform(0, 1)
        X = x_law.rvs(n * d).reshape(n, d)
        # print(X)
        y = generate_y(X, gen_f4, eps)
        print(y)
        x_train, x_test, y_train, y_test = train_test_split(X.astype(np.double), y.astype(np.double),
                                                            test_size=config.DataParams.test_ratio,
                                                            random_state=config.UtilsParams.seed)


    if name == 'generate_5_dim':
        eps_x = st.norm(0, 0.5)
        def gen_f5(X):

            return y
        n = 5000
        d = 50
        eps = st.norm(0, 1e-5)
        x_law = st.uniform(0, 1)
        X = x_law.rvs(n * d).reshape(n, d)
        # print(X)
        y = generate_y(X, gen_f5, eps)
        x_train, x_test, y_train, y_test = train_test_split(X.astype(np.double), y.astype(np.double),
                                                            test_size=config.DataParams.test_ratio,
                                                            random_state=config.UtilsParams.seed)

    return x_train, x_test, y_train, y_test

def gen_f1(X):
    x = X[0]
    if x >= 1.5 and x <= 2:
        return 1 + 0.05*np.random.randn(1)
    if x >= 4.5 and x <= 5:
        return 1 + 0.05*np.random.randn(1)
    return np.sin(x) ** 2 + 0.1 + 0.6 * np.sin(2 * x) * np.random.randn(1)


def generate_y(X, gen_y, eps):
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y[i] = gen_y(X[i]) + eps.rvs(1)
    return y


def compute_best_pi(X, gen_y, eps, pred, n=1000, quantile=0.9):
    r_mc = np.zeros(shape=(X.shape[0], n))
    for i in tqdm(range(n)):
        r_mc[:, i] = np.abs(generate_y(X, gen_y, eps) - pred)

    return np.quantile(r_mc, q=quantile, axis=1)

def cache(url, file_name=None):
    """ Loads a file from the URL and caches it locally.
    """
    if file_name is None:
        file_name = os.path.basename(url)
    data_dir = os.path.join(os.path.dirname(__file__), "cached_data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    file_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, file_path)

    return file_path