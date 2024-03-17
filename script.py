import os
import cooler
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import statsmodels.stats.multitest
from sklearn.cluster import DBSCAN, OPTICS, Birch

def get_qvalues(path_to_matrix, resolution, genome_position, end_bin, quantile_threshold = 0.9, fdr_correction = 0.5):
    mtx_name_for_cooler = path_to_matrix + '::/resolutions/' + str(resolution)
    balanced_matrix = cooler.Cooler(mtx_name_for_cooler).matrix(balance=True, sparse=False).fetch(genome_position)
    compare_dataframe = pd.DataFrame()
    for i in range(1, end_bin):
        diag = np.diag(balanced_matrix, k = i)
        diag_nan_omit = diag[~np.isnan(diag)]
        diag_temp = diag_nan_omit[diag_nan_omit > 0]
        diag_temp = diag_temp[diag_temp < np.quantile(diag_temp, quantile_threshold)]
        weibull_param = scipy.stats.weibull_min.fit(diag_temp, floc = 0)
        diag = np.nan_to_num(diag, nan = 0)
        diag_pval = 1 - (scipy.stats.weibull_min.cdf(diag, c = weibull_param[0],loc = weibull_param[1], scale = weibull_param[2]))
        diag_pval = statsmodels.stats.multitest.fdrcorrection(diag_pval, alpha = fdr_correction)[1]
        compare_dataframe = pd.concat([compare_dataframe, pd.DataFrame([diag_pval])], ignore_index = True, axis = 0)
    return compare_dataframe.T


def create_qval_log_mtx(qval_df, end_bin):
    temp_mtx = np.zeros((qval_df.shape[0] + 1,qval_df.shape[0] + 1))
    for i in range(1, end_bin):
        upper = qval_df[i - 1]
        np.fill_diagonal(temp_mtx[:-i, i:], upper)   
    #temp_mtx = temp_mtx + temp_mtx.T  
    log_qval = np.log10(np.array(temp_mtx))
    return log_qval


def get_dots_coords(qval_log_mtx):
    values = np.where((~np.isneginf(qval_log_mtx)) & (qval_log_mtx != 0) & (qval_log_mtx<=np.log10(0.01)))
    coords = pd.DataFrame()
    for i, j in zip(values[0], values[1]):
        new_row = [i, j, qval_log_mtx[i, j]]
        coords = pd.concat([coords, pd.DataFrame([new_row])], ignore_index=True)
    return coords
    
def draw_dots(coordinates):
    max_x = coordinates.iloc[:,0].max() + 1
    max_y = coordinates.iloc[:,1].max() + 1
    heatmap_data = np.zeros((int(max_x), int(max_y)))
    for _, row in coordinates.iterrows():
        heatmap_data[int(row[0]), int(row[1])] += row[2]
    plt.rcParams["figure.figsize"] = 7, 7
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.show()


def main_func(path_to_matrix, resolution, genome_position, end_bin, quantile_threshold = 0.9, fdr_correction = 0.5):
    qval_dataframe = get_qvalues(path_to_matrix, resolution, genome_position, end_bin)
    np.seterr(divide='ignore')
    log_matrix = create_qval_log_mtx(qval_dataframe, end_bin)
    sign_dots = get_dots_coords(log_matrix)
    draw_dots(sign_dots)
    return sign_dots

    #fig = plt.figure(figsize=(7, 7))
    #plt.imshow(log_matrix, cmap='afmhot_r')

    #return log_matrix