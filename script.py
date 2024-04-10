import os
import warnings

import cooler
import cooltools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import statsmodels.stats.multitest
from coolpuppy import coolpup, plotpup
from sklearn.cluster import DBSCAN, OPTICS

warnings.simplefilter(action='ignore')

def get_qvalues(path_to_matrix, resolution, genome_position, end_bin, start_bin, quantile_threshold, fdr_correction):
    mtx_name_for_cooler = path_to_matrix + '::/resolutions/' + str(resolution)
    balanced_matrix = cooler.Cooler(mtx_name_for_cooler).matrix(balance=True, sparse=False).fetch(genome_position)
    compare_dataframe = pd.DataFrame()
    for i in range(start_bin, end_bin):
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
    log_qval = np.log10(np.array(temp_mtx))
    return log_qval


def get_dots_coords(qval_log_mtx, qval_threshold):
    values = np.where((~np.isneginf(qval_log_mtx)) & (qval_log_mtx != 0) & (qval_log_mtx<=np.log10(qval_threshold)))
    coords = pd.DataFrame()
    for i, j in zip(values[0], values[1]):
        new_row = [i, j, qval_log_mtx[i, j]]
        coords = pd.concat([coords, pd.DataFrame([new_row])], ignore_index=True)
    col_names = ['start', 'end', 'qval']
    coords = coords.set_axis(col_names, axis=1)
    return coords


def dbscan_clustering(coordinates, eps, min_samples, genome_position, resolution):
    coor = coordinates[['start', 'end']]
    clusterer = DBSCAN(eps=eps, min_samples = min_samples)
    cluster_labels_all_db = clusterer.fit_predict(coor)
    coor['dbscan'] = cluster_labels_all_db
    coor_dbscan = coor.query('dbscan != -1')
    loops_coords_bed = pd.DataFrame()
    labels = coor_dbscan['dbscan'].unique()
    for i in labels:
        new_row = round(coor_dbscan.query('dbscan == @i')[{'start', 'end'}].apply(np.mean, axis=0)).astype(int)
        loops_coords_bed = pd.concat([loops_coords_bed, pd.DataFrame([new_row])], ignore_index=True)
    chrom_name = genome_position if ':' not in genome_position else genome_position.split(':')[0]
    loops_coords_bed.insert(0,'chrom','')
    loops_coords_bed['chrom'] = chrom_name
    loops_coords_bed[['start', 'end']] = loops_coords_bed[['start', 'end']].apply(lambda x: x * resolution)
    return loops_coords_bed


def optics_clusterig(coordinates, min_samples, max_eps, genome_position, resolution):
    coor = coordinates[['start', 'end']]
    optics = OPTICS(min_samples = min_samples, max_eps = max_eps)
    optics_pred = optics.fit_predict(coor)
    coor['optics'] = optics_pred
    coor_optics = coor.query('optics != -1')
    loops_coords_bed = pd.DataFrame()
    optics_labels = coor_optics['optics'].unique()
    for i in optics_labels:
        new_row = round(coor_optics.query('optics == @i')[{'start', 'end'}].apply(np.mean, axis=0)).astype(int)
        loops_coords_bed = pd.concat([loops_coords_bed, pd.DataFrame([new_row])], ignore_index=True)
    chrom_name = genome_position if ':' not in genome_position else genome_position.split(':')[0]
    loops_coords_bed.insert(0,'chrom','')
    loops_coords_bed['chrom'] = chrom_name
    loops_coords_bed[['start', 'end']] = loops_coords_bed[['start', 'end']].apply(lambda x: x * resolution)
    return loops_coords_bed
    

def pileup_dots(looop_genome_coords, path_to_matrix, resolution):
    mtx_name_for_cooler = path_to_matrix + '::/resolutions/' + str(resolution)
    hic = cooler.Cooler(mtx_name_for_cooler)
    expected = cooltools.expected_cis(hic, ignore_diags=0, chunksize=1000000)
    puppy = coolpup.pileup(hic, looop_genome_coords, features_format='bed', expected_df=expected, local=True, nshifts=10, flank=30_000, min_diag=0)
    pile_loops = plotpup.plot(puppy,score=False, cmap='coolwarm', scale='log', sym=True, vmax=2,height=5, plot_ticks=True)
    return pile_loops


def main_func(path_to_matrix, resolution, genome_position, end_bin, cluster_method = 'dbscan', start_bin=1, quantile_threshold = 0.9, fdr_correction = 0.5, qval_threshold = 0.01, eps=1, min_samples = 3, max_eps=1.5):
    qval_dataframe = get_qvalues(path_to_matrix, resolution, genome_position, end_bin, start_bin, quantile_threshold, fdr_correction)
    np.seterr(divide='ignore')
    log_matrix = create_qval_log_mtx(qval_dataframe, end_bin)
    sign_dots = get_dots_coords(log_matrix, qval_threshold)
    if cluster_method == 'dbscan':
        loops_genome_coords = dbscan_clustering(sign_dots, eps, min_samples, genome_position, resolution)
    elif cluster_method == 'optics':
        loops_genome_coords = optics_clusterig(sign_dots, min_samples, max_eps, genome_position, resolution)
    draw_average_loops = pileup_dots(loops_genome_coords, path_to_matrix, resolution)
    return draw_average_loops, loops_genome_coords
