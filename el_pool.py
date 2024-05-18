import os
import warnings
import time
import logging

import cooler
import cooltools
import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
import matplotlib.pyplot as plt
from coolpuppy import coolpup, plotpup
from sklearn.cluster import OPTICS
from typing import Union

warnings.simplefilter(action = 'ignore')
np.seterr(divide = 'ignore')
logger = logging.getLogger('Elon_MaskED')

def get_qvalues(path_to_matrix: str, resolution: int, genome_position: str, end_bin: int, start_bin: int,
                quantile_threshold: float, fdr_correction: float) -> pd.DataFrame:
    """
    Calculates Weibull distribution q-values for each pixel in region of interest from diagonal k (start bin) to n (end bin)

    Input arguments:
    - path_to_matrix (str): path to mcool/cool HiC matrix
    - resolution (int): resolution of HiC matrix
    - genome_position (str): chromosome or specific genome region in UCSC format i.g. chr3:10000-20000
    - start bin (int): left boudary of region of interest, the higher the value, the further away from the main diagonal
    - end bin (int): right boudary of region of interest 
    - quantile_threshold (float): the threshold for filtering out the bright outlier pixels
    - fdr_correction (float): significants threshold to correct for multiple comparisons

    Output:
    pd.DataFrame with q-values for each pixel in region of interest
    """
    mtx_name_for_cooler = path_to_matrix + '::/resolutions/' + str(resolution)
    balanced_matrix = cooler.Cooler(mtx_name_for_cooler).matrix(balance=True, sparse=False).fetch(genome_position)
    q_values = pd.DataFrame()
    for i in range(start_bin, end_bin):
        diag = np.diag(balanced_matrix, k = i)
        diag_nan_omit = diag[~np.isnan(diag)]
        diag_temp = diag_nan_omit[diag_nan_omit > 0]
        diag_temp = diag_temp[diag_temp < np.quantile(diag_temp, quantile_threshold)]
        weibull_param = scipy.stats.weibull_min.fit(diag_temp, floc = 0)
        diag = np.nan_to_num(diag, nan = 0)
        diag_pval = 1 - (scipy.stats.weibull_min.cdf(diag, c = weibull_param[0],loc = weibull_param[1], scale = weibull_param[2]))
        diag_pval = statsmodels.stats.multitest.fdrcorrection(diag_pval, alpha = fdr_correction)[1]
        q_values = pd.concat([q_values, pd.DataFrame([diag_pval])], ignore_index = True, axis = 0)
    q_values = q_values.T
    return q_values


def create_qval_log_mtx(qval_df: pd.DataFrame, end_bin: int) -> np.array:
    """
    Converts q values dataframe to logarithmic scale

    Input arguments:
    - qval_df (pd.DataFrame): q values data about each pixel
    - end bin (int): right boudary of region of interest 

    Output:
    np.array with log q values
    """
    temp_mtx = np.zeros((qval_df.shape[0] + 1, qval_df.shape[0] + 1))
    for i in range(1, end_bin):
        upper = qval_df[i - 1]
        np.fill_diagonal(temp_mtx[:-i, i:], upper)    
    log_qval = np.log10(np.array(temp_mtx))
    return log_qval


def get_dots_coords(qval_log_mtx: np.array, qval_threshold: float) -> pd.DataFrame:
    """
    Filters out log q values and get bin coordinates.

    Input arguments:
    - qval_log_mtx (np.array): q values for each pixel in logarithmic scale
    - qval_threshold (float): threshold to filter q values, the higher the softer the threshold value

    Output:
    pd.DataFrame with bin coordinates
    """
    values = np.where((~np.isneginf(qval_log_mtx)) & (qval_log_mtx != 0) & (qval_log_mtx<=np.log10(qval_threshold)))
    coords = pd.DataFrame()
    for i, j in zip(values[0], values[1]):
        new_row = [i, j, qval_log_mtx[i, j]]
        coords = pd.concat([coords, pd.DataFrame([new_row])], ignore_index=True)
    col_names = ['start', 'end', 'qval']
    coords = coords.set_axis(col_names, axis = 1)
    return coords


def optics_clustering(coordinates: pd.DataFrame, min_samples: int, max_eps: float, 
                     genome_position: str, resolution: int) -> pd.DataFrame:
    """
    Clusters filtered pixels using OPTICS algorithm

    Input arguments:
    - coordinates (pd.DataFrame): pd.DataFrame with bin coordinates for each log q value
    - min_samples (int): number of samples in a neighborhood for a pixel
    - max_eps (float): maximum distance between two samples for one
    - genome_position (str): chromosome or specific genome region in UCSC format i.g. chr3:10000-20000
    - resolution (int): resolution of HiC matrix

    Output:
    pd.DataFrame in bedpe-like format with centroid coordinated of each defined cluster
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
    loops_coords_bedpe = loops_coords_bedpe.drop(['start', 'end'], axis = 1)
    loops_coords_bedpe = loops_coords_bedpe.query('start1 >= 100000')
    return loops_coords_bedpe
    

def pileup_dots(loop_coords: pd.DataFrame, path_to_matrix: str, resolution: int, elong: Union[str | None], 
                visualization: bool = False) -> pd.DataFrame:
    """ 
    Creates pileup of dots ("mean loop") with coolpuppy, creates pileup figure

    Input arguments:
    - loop_coords (pd.DataFrame): bedpe-like format with centroid coordinated of each defined cluster
    - path_to_matrix (str): path to mcool/cool HiC matrix
    - resolution (int): resolution of HiC matrix
    - elong (str or None): loop orientation: left or right, None for all type of loops
    - visualization (bool): if True pileup figure will be created

    Ouput:
    pd.DataFrame with results of coolpuppy run
    """
    logger.info('Run pileup')
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
    expected = cooltools.expected_cis(hic, chunksize = 1000000, view_df = view)
    puppy = coolpup.pileup(clr = hic, features = loop_coords, view_df = view, features_format = 'bedpe', expected_df = expected, 
                            nshifts = 100000, flank = 8500)
    if visualization:
        pile_loops = plotpup.plot(puppy, score = True, cmap = 'coolwarm', scale = 'log', sym = True, 
                                  vmax = 2.1, height = 5, plot_ticks = True)
        pile_loops.savefig(f'pileup_{elong}.pdf', bbox_inches='tight')
        plt.close(pile_loops.fig)
    return puppy


def create_mask(puppy: pd.DataFrame, elong: Union[str | None]) -> np.array:
    """
    Creates mask for loop searching, elongated or summetrical

    Input arguments:
    - puppy (pd.DataFrame): results of coolpuppy run
    - elong (str or None): loop orientation: left or right, None for all type of loops

    Output:
    np.array - mask for loop searching
    """
    puppy = puppy.data[0]
    peak = puppy.max()
    lowest_value = peak - puppy.min()
    itemindex = np.where(puppy == peak)
    row = itemindex[0][0]
    column = itemindex[1][0]
    row_iter = row
    column_iter = column
    while puppy[row_iter][column] > lowest_value and (row_iter < puppy.shape[0] - 1):
        row_iter = row_iter + 1
    value = puppy[row_iter - 1][column]
    item_row = np.where(puppy == value)
    while (puppy[row][column_iter] > lowest_value) and (column_iter < puppy.shape[0] - 1):
        column_iter = column_iter + 1
    value = puppy[row][column_iter - 2]
    item_column = np.where(puppy == value)
    row_end = item_row[0][0]
    column_end = item_column[1][0]
    step_row = row_end - row
    step_column = column_end - column
    pp_mask = puppy[row - step_row : row + step_row, column - step_column : column + step_column]
    pp_mask = pp_mask * -1
    if pp_mask.shape != (2,2):
        pp_mask = np.random.choice(pp_mask.flatten(), (2,2))
    if elong == 'left':
        pp_mask = np.repeat(pp_mask, 2, axis = 1)
        pp_mask = np.pad(pp_mask, [(4, ), (3, )], mode='constant')
    elif elong == 'right':
        pp_mask = np.repeat(pp_mask, 2, axis = 0)
        pp_mask = np.pad(pp_mask, [(3, ), (4, )], mode='constant')
    elif elong is None:
        pp_mask = np.repeat(np.repeat(pp_mask, 2, axis = 0), 2, axis = 1)
        pp_mask = np.pad(pp_mask, [(3, ), (3, )], mode='constant')
    return pp_mask

def count_mask_vs_loop_corelation(pp_mask: np.array, coords: pd.DataFrame, log_mtx: np.array) -> pd.DataFrame:
    """
    Counts Spearmann correlation between potential loop and mask

    Input arguments:
    - pp_mask (np.array): mask for loop searching
    - coords (pd.DataFrame): bedpe-like format with centroid coordinated of each defined cluster
    - log_mtx (np.array): log q values

    Output:
    pd.DataFrame in bedpe-like format with filtered loop coordinates
    """
    dict_for_coeffs = {}
    threshold = 0.5
    for i in range(len(coords)):
        coordinates_dots = coords.iloc[i][['start1', 'start2', 'end1', 'end2']].apply(lambda x: x // 2000)
        start_1, start_2, end_1, end_2 = [j for j in coordinates_dots]
        mtx = log_mtx[start_1 : end_1, start_2 : end_2]
        if mtx.shape != pp_mask.shape:
            continue
        mtx = np.nan_to_num(mtx, neginf = 0) 
        candidate_loop = mtx.flatten()
        mask = pp_mask.flatten()
        coeff = scipy.stats.spearmanr(candidate_loop, mask)[0]
        if coeff >= threshold:
            dict_for_coeffs[i] = scipy.stats.spearmanr(candidate_loop, mask)[0]
    coords_elog = pd.DataFrame()
    for i in dict_for_coeffs:
        row = list(coords.iloc[i])
        row = pd.DataFrame([row])
        coords_elog = pd.concat([coords_elog, row])
    col_names = ['chrom1', 'start1', 'start2', 'chrom2', 'end1', 'end2']
    coords_elog = coords_elog.set_axis(col_names, axis = 1)
    return coords_elog


def write_bedpe(coords_elog: pd.DataFrame, elong: Union[str | None], path: str, genome_position: str) -> None:
    """
    Save bedpe file with loop coordinates in working directory

    Input arguments:
    - coords_elog (pd.DataFrame): bedpe-like format with filtered loop coordinates
    - elong (str or None): loop orientation: left or right, None for all type of loops
    - path (str): path to mcool/cool HiC matrix
    - genome_position (str): chromosome or specific genome region in UCSC format i.g. chr3:10000-20000

    Output:
    Save bedpe file, return None
    """
    coords_elog.name = None
    path_name = os.path.basename(path).split('.')[0]
    file_name = path_name + f'_{genome_position}_orient_{elong}.bedpe'
    coords_elog.to_csv(file_name, header = False, index = False, sep = '\t')
    return None


def run_elong_loop_caller(path_to_matrix: str, resolution: int, genome_position: str, end_bin: int, start_bin: int = 1, 
                          quantile_threshold: float = 0.9, fdr_correction: float = 0.5, qval_threshold: float = 0.01, 
                          min_samples: int = 3, max_eps: float = 1.5, elong: Union[str | None] = 'left') -> None:
    """
    Detects elongated loops using mask-based approach. HiC pixel values are filtered out, clustered using OPTICS
    and extract elongated loops comparing the correlation value between the mask and the loop with the threshold value.

    Arguments:
    - path_to_matrix (str): path to mcool/cool HiC matrix
    - resolution (int): resolution of HiC matrix
    - genome_position (str): chromosome or specific genome region in UCSC format i.g. chr3:10000-20000
    - start bin (int): left boudary of region of interest, the higher the value, the further away from the main diagonal,
      default 1
    - end bin (int): right boudary of region of interest 
    - quantile_threshold (float): the threshold for filtering out the bright outlier pixels, default 0.9
    - fdr_correction (float): significants threshold to correct for multiple comparisons, default 0.5
    - qval_threshold (float): threshold to filter q values, the higher the softer the threshold value, default 0.01
    - min_samples (int): number of samples in a neighborhood for a point to be considered as a core point, default 3
    - max_eps (float): the maximum distance between two samples for one to be considered as in the neighborhood of the other,
      default 1.5
    - (str or None): loop orientation: left or right, None for all type of loops

    Output:
    Function saves bedpe file with coordinates and pileup figures.
    """
    start_time = time.perf_counter()
    logger.info('Start Elon MaskED')
    qval_dataframe = get_qvalues(path_to_matrix, resolution, genome_position, end_bin, start_bin, 
                                 quantile_threshold, fdr_correction)
    log_matrix = create_qval_log_mtx(qval_dataframe, end_bin)
    sign_dots = get_dots_coords(log_matrix, qval_threshold)
    loops_genome_coords = optics_clustering(sign_dots, min_samples, max_eps, genome_position, resolution)
    puppy_data = pileup_dots(loops_genome_coords, path_to_matrix, resolution, elong)
    mask = create_mask(puppy_data, elong)
    elong_loop_df = count_mask_vs_loop_corelation(mask, loops_genome_coords, log_matrix)
    pileup_dots(elong_loop_df, path_to_matrix, resolution, elong, visualization=True)
    logger.info('Saving results')
    write_bedpe(elong_loop_df, elong, path_to_matrix, genome_position)
    end_time = time.perf_counter()
    logger.info(f'Done! Running time is {(end_time - start_time):.02f} sec')
    return None
