import os
import warnings

import cooler
import cooltools
import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
from coolpuppy import coolpup, plotpup
from sklearn.cluster import OPTICS

warnings.simplefilter(action='ignore')
np.seterr(divide='ignore')

def get_qvalues(path_to_matrix, resolution, genome_position, end_bin, start_bin, quantile_threshold, fdr_correction):
    """
    """
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
    """
    """
    temp_mtx = np.zeros((qval_df.shape[0] + 1,qval_df.shape[0] + 1))
    for i in range(1, end_bin):
        upper = qval_df[i - 1]
        np.fill_diagonal(temp_mtx[:-i, i:], upper)    
    log_qval = np.log10(np.array(temp_mtx))
    return log_qval


def get_dots_coords(qval_log_mtx, qval_threshold):
    """
    """
    values = np.where((~np.isneginf(qval_log_mtx)) & (qval_log_mtx != 0) & (qval_log_mtx<=np.log10(qval_threshold)))
    coords = pd.DataFrame()
    for i, j in zip(values[0], values[1]):
        new_row = [i, j, qval_log_mtx[i, j]]
        coords = pd.concat([coords, pd.DataFrame([new_row])], ignore_index=True)
    col_names = ['start', 'end', 'qval']
    coords = coords.set_axis(col_names, axis=1)
    return coords


def optics_clusterig(coordinates, min_samples, max_eps, genome_position, resolution):
    """
    """
    coor = coordinates[['start', 'end']]
    optics = OPTICS(min_samples = min_samples, max_eps = max_eps)
    optics_pred = optics.fit_predict(coor)
    coor['optics'] = optics_pred
    coor_optics = coor.query('optics != -1')
    loops_coords_bedpe = pd.DataFrame()
    optics_labels = coor_optics['optics'].unique()
    for i in optics_labels:
        new_row = round(coor_optics.query('optics == @i')[{'start', 'end'}].apply(np.mean, axis=0)).astype(int)
        loops_coords_bedpe = pd.concat([loops_coords_bedpe, pd.DataFrame([new_row])], ignore_index=True)
    chrom_name = genome_position if ':' not in genome_position else genome_position.split(':')[0]
    loops_coords_bedpe['chrom1'] = chrom_name
    loops_coords_bedpe[['start1', 'start2']] = loops_coords_bedpe[['start', 'end']].apply(lambda x: (x - 5) * resolution)
    loops_coords_bedpe['chrom2'] = chrom_name
    loops_coords_bedpe[['end1', 'end2']] = loops_coords_bedpe[['start', 'end']].apply(lambda x: (x + 5) * resolution)
    loops_coords_bedpe = loops_coords_bedpe.drop(['start', 'end'], axis=1)
    loops_coords_bedpe = loops_coords_bedpe.query('start1 >= 300000')
    return loops_coords_bedpe
    

def pileup_dots(loop_coords, path_to_matrix, resolution, visualization=False):
    """
    """
    mtx_name_for_cooler = path_to_matrix + '::/resolutions/' + str(resolution)
    hic = cooler.Cooler(mtx_name_for_cooler)
    sizes = hic.chromsizes
    sizes = sizes.to_frame()
    sizes = sizes.reset_index()
    view = pd.DataFrame()
    view['chrom'] = sizes['name']
    view['start'] = 0
    view['end'] = sizes['length']
    view['name'] = sizes['name']
    expected = cooltools.expected_cis(hic, chunksize=1000000, view_df=view)
    puppy = coolpup.pileup(clr=hic, features=loop_coords, view_df=view, features_format='bedpe', expected_df=expected, 
                            nshifts=100000, flank=8500)
    if visualization:
        pile_loops = plotpup.plot(puppy, score=True, cmap='coolwarm', scale='log', sym=True, vmax=2.5, height=5, plot_ticks=False)
    return puppy


def create_mask(puppy, elong):
    """
    """
    puppy = puppy.data[0]
    peak = puppy.max()
    lowest_value = peak - puppy.min()
    itemindex = np.where(puppy == peak)
    row = itemindex[0][0]
    column = itemindex[1][0]
    row_iter = row
    column_iter = column
    while puppy[row_iter][column] > lowest_value and (row_iter < puppy.shape[0]-1):
        row_iter = row_iter+1
    value = puppy[row_iter-1][column]
    item_row = np.where(puppy == value)
    while (puppy[row][column_iter] > lowest_value) and (column_iter < puppy.shape[0]-1):
        column_iter = column_iter+1
    value = puppy[row][column_iter-2]
    item_column = np.where(puppy == value)
    end1 = item_row[0][0]
    end2 = item_column[1][0]
    vl = end1 - row
    vl_2 = end2 - column
    pp = puppy[row - vl:row+vl, column - vl_2 :column+vl_2]
    pp = pp * -1
    if pp.shape != (2,2):
        pp = np.random.choice(pp.flatten(), (2,2))
    if elong == 'left':
        pp = np.repeat(pp, 2, axis=1)
        pp = np.pad(pp, [(4, ), (3, )], mode='constant')
    elif elong == 'right':
        pp = np.repeat(pp,2, axis=0)
        pp = np.pad(pp, [(3, ), (4, )], mode='constant')
    elif elong is None:
        pp = np.repeat(np.repeat(pp,2, axis=0), 2, axis=1)
        pp = np.pad(pp, [(3, ), (3, )], mode='constant')
    return pp


def run_elong_loop_caller(path_to_matrix, resolution, genome_position, end_bin, start_bin=1, quantile_threshold = 0.9, fdr_correction = 0.5, qval_threshold = 0.01, min_samples = 3, max_eps=1.5, elong='left'):
    """
    """
    qval_dataframe = get_qvalues(path_to_matrix, resolution, genome_position, end_bin, start_bin, quantile_threshold, fdr_correction)
    log_matrix = create_qval_log_mtx(qval_dataframe, end_bin)
    sign_dots = get_dots_coords(log_matrix, qval_threshold)
    loops_genome_coords = optics_clusterig(sign_dots, min_samples, max_eps, genome_position, resolution)
    puppy_data = pileup_dots(loops_genome_coords, path_to_matrix, resolution)
    mask = create_mask(puppy_data, elong)
    return mask
