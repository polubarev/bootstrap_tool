from tqdm import tqdm

import numpy as np

from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest

from joblib import Parallel, delayed

def bootstrap_metric(data, metric='mean', n_samples=None, iters=5000):
    if n_samples is None:
        n_samples = len(data)
        
    if metric == 'mean':
        return np.random.choice(data, (iters, n_samples)).mean(axis=1)
    
    elif metric == 'median':
        return np.median(np.random.choice(data, (iters, n_samples)), axis=1)

    elif metric == 'std':
        return np.random.choice(data, (iters, n_samples)).std(axis=1)
    
    else:
        raise ValueError("Wrong metric") 

def diff_CI(data1, data2, metric='mean', confidence=[.025, .975], add_obs=0, iters=5000):
    boot_1 = bootstrap_metric(data1, metric=metric, n_samples=len(data1) + add_obs, iters=iters)
    boot_2 = bootstrap_metric(data2, metric=metric, n_samples=len(data2) + add_obs, iters=iters)
    
    abs_diff = boot_2 - boot_1
    rel_diff = boot_2 / boot_1 - 1
    
    return np.quantile(abs_diff, confidence),  np.quantile(rel_diff, confidence)


def bootstrap_test(data1, data2, metric='mean', add_obs=0, iters=5000):
    new_data = np.concatenate([data1, data2])
    
    boot_1 = bootstrap_metric(new_data, metric=metric, n_samples=len(data1) + add_obs, iters=iters)
    boot_2 = bootstrap_metric(new_data, metric=metric, n_samples=len(data2) + add_obs, iters=iters)
    
    abs_diff = boot_2 - boot_1
    
    if metric == 'mean':
        obs_diff = data1.mean() - data2.mean()
    
    elif metric == 'median':
        obs_diff = data1.median() - data2.median()

    elif metric == 'std':
        obs_diff = data1.std() - data2.std()
    
    else:
        raise ValueError("Wrong metric") 

    return (abs(abs_diff) >= abs(obs_diff)).mean()


def power(data1, data2, alpha=.05, test_type='boot', simulations=500, boot_iters=5000, add_obs=0, n_jobs=-1, tqdm_disable=True):
    
    # inner cycle for calculation paralleling
    def inner_cycle(data1, data2, test_type, boot_iters, add_obs):
        
        control = np.random.choice(data1, len(data1) + add_obs)
        test = np.random.choice(data2, len(data2) + add_obs)
        
        if test_type == 'boot':
            p_value = bootstrap_test(control, test, iters=boot_iters, add_obs=0)
            
        elif test_type == 'ztest':
            p_value = ztest(control, test)[1]
        
        elif test_type == 'ttest':
            p_value = ttest_ind(control, test)[1]
        
        else:
            return print('wrong test type')
    
        return p_value
    
    # симуляции тестов
    sim_res_boot = Parallel(n_jobs=-1)(delayed(inner_cycle)(data1, data2, test_type, boot_iters, add_obs) for _ in tqdm(range(simulations), disable=tqdm_disable))
    
    # подсчет количества тестов показавших значимость
    sim_res_boot = np.array(sim_res_boot)
    res_boot = np.sum((sim_res_boot <= alpha)) / simulations
    
    return res_boot