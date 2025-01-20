import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import glob
import pickle
import itertools
import math
import heapq
import datetime
import multiprocessing
from scipy.linalg import pinv
import time
#from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import random
from joblib import Parallel, delayed
from scipy.optimize import brute, differential_evolution, dual_annealing, shgo
import itertools
from functools import partial
from multiprocessing import Pool
from concurrent import futures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold
from config import *
from optimisation_helicoid import *
import mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from concurrent.futures import ProcessPoolExecutor

from optimisation import plot_concentrations #change the plot_concentrations function in optimisation.py
from optimisation import preprocess_piglet
from optimisation import dic, get_spectra, y_hbo2_f, y_hb_f, y_coxa, y_creda

from tqdm.notebook import tqdm

path_wlopti = "./wlopti/"
if not os.path.exists(path_wlopti):
    os.makedirs(path_wlopti)
    
def get_piglet_spectra(left_cut, right_cut):
    piglet_train_set = list(filter(lambda x: x not in ['LWP503','LWP504','LWP507', 'LWP509','LWP511','LWP512'], list(dic.keys())))
    print(piglet_train_set)
    dataset_spectr = np.array([])
    for piglet in piglet_train_set:
        wavelengths, M, comp_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_spectra(piglet, left_cut, right_cut)
        dataset_spectr = np.concatenate((dataset_spectr, comp_spectr), axis=0) if dataset_spectr.size else comp_spectr
    return wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda

def get_piglet_spectra_light(left_cut, right_cut, coarseness_wl=5, coarseness_data=100):
    piglet_train_set = list(filter(lambda x: x not in ['LWP503','LWP504','LWP507', 'LWP509','LWP511','LWP512'], list(dic.keys())))
    print(piglet_train_set)
    dataset_spectr = np.array([])
    for piglet in piglet_train_set:
        wavelengths, M, comp_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_spectra(piglet, left_cut, right_cut)
        dataset_spectr = np.concatenate((dataset_spectr, comp_spectr[::coarseness_data,::coarseness_wl]), axis=0) if dataset_spectr.size else comp_spectr[::coarseness_data,::coarseness_wl]
    return wavelengths[::coarseness_wl], M[::coarseness_wl,:], dataset_spectr, y_hbo2_f[::coarseness_wl], y_hb_f[::coarseness_wl], y_coxa[::coarseness_wl], y_creda[::coarseness_wl]

# def calculate_mse(i, wavelength_index_combinations, M, dataset_spectr, concentrations_all_wavelengths):
#     indices = wavelength_index_combinations[i]
#     concentrations = (pinv(M[indices,:]) @ dataset_spectr[:,indices].T).T
#     return np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))

opti_patients = ["012-01","020-01","008-02", "012-02", "015-01", "008-01", "016-04", "016-05", "025-02"]


#Specificy path to Helicoid, UCL-NIR-Spectra, CREATIS Spectra
dataset_path = "./dataset/"
path_absorp = "./dataset/UCL-NIR-Spectra/spectra/"
path_creatis = "./dataset/CREATIS-Spectra/spectra/"

from spectral import * 
spectral.settings.envi_support_nonlowercase_params = True
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bisect
import imageio as io
from PIL import Image
import numpy as np
import scipy
import random
import os
from tqdm import tqdm
from scipy.linalg import pinv
from utils import save_helicoid_optimization_data
import time
from config import left_cut_helicoid, right_cut_helicoid
from optimisation_helicoid import *

# def helicoid_run_with_wl(wl, only_labeled=False): #contains wavelengths to be used
#     path = path_helicoid
#     for patient in ["008-02", "025-02", "016-04", "016-05", "020-01"]:
#         hdr_path = path+"/{}/raw.hdr".format(patient)
#         img = open_image(hdr_path)
#         wavelength = np.array(img.metadata['wavelength']).astype(float)

#         rbg_path = path+"/{}/image.jpg".format(patient)
#         rbg = Image.open(rbg_path)

#         gt_path = path+"/{}/gtMap.hdr".format(patient)
#         gt = open_image(gt_path)
#         gt = gt.load()
#         #print(gt.shape)

#         mpl.rc('image', cmap='terrain')

#         mpl.rcParams['text.color'] = 'grey'
#         mpl.rcParams['xtick.color'] = 'grey'
#         mpl.rcParams['ytick.color'] = 'grey'
#         mpl.rcParams['axes.labelcolor'] = 'grey'

#         tumor = gt==2
#         normal = gt==1
#         blood = gt==3
#         mask1 = np.ma.masked_where(tumor.astype(int) == 0, tumor.astype(int))
#         mask2 = np.ma.masked_where(normal.astype(int) == 0, normal.astype(int))
#         mask3 = np.ma.masked_where(blood.astype(int) == 0, blood.astype(int))

#         white_path = path+"/{}/whiteReference.hdr".format(patient)
#         white = open_image(white_path)
#         white = white.load()

#         dark_path = path+"/{}/darkReference.hdr".format(patient)

#         dark = open_image(dark_path)
#         dark = dark.load()
#         white_full = np.tile(white, (img.shape[0],1,1))
#         white_full_RGB = np.stack((white_full[:,:,int(img.metadata['default bands'][2])],white_full[:,:,int(img.metadata['default bands'][1])],white_full[:,:,int(img.metadata['default bands'][0])]), axis=2)
#         dark_full = np.tile(dark, (img.shape[0],1,1))

#         #img_normalized = np.array(img.load())
#         #img_normalized = (img.load() - dark_full) / (white_full - dark_full)
#         #img_normalized = img_normalized * 2
#         img_normalized = sfilter((img.load() - dark_full) / (white_full - dark_full)) + 0.1
#         #img_normalized = shiftw(img_normalized)
#         #img_normalized = np.array(img.load())
#         img_normalized[img_normalized <= 0] = 10**-2
#         #print(img_normalized.shape)
#         ### Visualising spectrograms

#         #intensity1 = []
#         #intensity2 = []

#         x_blood,y_blood,z_blood = np.where(blood==1)
#         #blood_index = 5
#         #orange_dot = [x_blood[blood_index], y_blood[blood_index]]

#         #blue_dot = [110,180]
#         #orange_dot = [220,160]
#         #orange_dot = [82,267]
#         #wavelength = np.linspace(400, 1000, 826)
#         wavelength = np.array(img.metadata['wavelength']).astype(float)
#         wavelength_filtered = wavelength

#         img_RGB = np.stack((img_normalized[:,:,int(img.metadata['default bands'][2])],img_normalized[:,:,int(img.metadata['default bands'][1])],img_normalized[:,:,int(img.metadata['default bands'][0])]), axis=2)

#         mol_list = ["water_hale"]
#         molecules_ucl, x_ucl = read_molecules(left_cut, right_cut, mol_list, wavelength_filtered)

#         molecules_cyto, x_cyto = read_molecules_cytochrome_cb(left_cut, right_cut, wavelength_filtered)

#         molecules, x = read_molecules_creatis(left_cut, right_cut, x_waves=wavelength_filtered)

#         wavelengths_chosen = np.isin(wavelength_filtered, wl)
#         x_chosen = np.isin(x, wl)

#         y_c_oxy, y_c_red, y_b_oxy, y_b_red = molecules_cyto #, y_water, y_fat = molecules

#         y_hb_f, y_hbo2_f, y_coxa, y_creda, y_fat, y_water = molecules

#         M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
#                                     np.asarray(y_hb_f),
#                                     np.asarray(y_coxa),
#                                     np.asarray(y_creda),
#                                     np.asarray(y_c_oxy),
#                                     np.asarray(y_c_red),
#                                     np.asarray(y_b_oxy),
#                                     np.asarray(y_b_red),                                    
#                                     np.asarray(y_water),
#                                     np.asarray(y_fat))))

#         M = M[x_chosen,:]
#         img_normalized = img_normalized[:,:,wavelengths_chosen]

#         config.molecule_count = M.shape[1]
#         config.max_b = 4
#         config.max_a = 100
#         coarseness = 1
#         hbt_opti = np.zeros((img.shape[0], img.shape[1]))
#         hbdiff_opti = np.zeros((img.shape[0], img.shape[1]))
#         error_opti = np.zeros((img.shape[0], img.shape[1]))
#         diffCCO_opti = np.zeros((img.shape[0], img.shape[1]))

#         #reference_index_blood = random.randint(0, len(x_blood) - 1)
#         #reference_point = [x_blood[reference_index_blood], y_blood[reference_index_blood]]
#         reference_point = torch.load("dataset/helicoid/" + patient + "/reference_point.pt")
#         #coefs, scattering_params, errors, a_t1, b_t1, img_ref = helicoid_optimisation_scattering_wlselection(img_normalized, reference_point, M, x[x_chosen], 8, patient, x_chosen)
        
#         error_wavelength_selection, params_found, params_found_GT = helicoid_optimisation_scattering_wlselection(img_normalized, reference_point, M, x[x_chosen], 3, patient, x_chosen, label_map=gt, only_labeled=only_labeled)
        
#         return error_wavelength_selection, params_found, params_found_GT 

def helicoid_run_with_wl(wl, only_labeled=False, coarseness=6, use_parallel_lsq=False): #contains wavelengths to be used
    path = path_helicoid
    
    error_wavelength_selection_patients = []
    params_found_patients = []
    params_found_GT_patients = [None, None, None, None, None]
    
    for patient_id, patient in enumerate(["008-02", "025-02", "016-04", "016-05", "020-01"]):
        hdr_path = path+"/{}/raw.hdr".format(patient)
        img = open_image(hdr_path)
        wavelength = np.array(img.metadata['wavelength']).astype(float)

        rbg_path = path+"/{}/image.jpg".format(patient)
        rbg = Image.open(rbg_path)

        gt_path = path+"/{}/gtMap.hdr".format(patient)
        gt = open_image(gt_path)
        gt = gt.load()
        #print(gt.shape)

        mpl.rc('image', cmap='terrain')

        mpl.rcParams['text.color'] = 'grey'
        mpl.rcParams['xtick.color'] = 'grey'
        mpl.rcParams['ytick.color'] = 'grey'
        mpl.rcParams['axes.labelcolor'] = 'grey'

        tumor = gt==2
        normal = gt==1
        blood = gt==3
        mask1 = np.ma.masked_where(tumor.astype(int) == 0, tumor.astype(int))
        mask2 = np.ma.masked_where(normal.astype(int) == 0, normal.astype(int))
        mask3 = np.ma.masked_where(blood.astype(int) == 0, blood.astype(int))

        white_path = path+"/{}/whiteReference.hdr".format(patient)
        white = open_image(white_path)
        white = white.load()

        dark_path = path+"/{}/darkReference.hdr".format(patient)

        dark = open_image(dark_path)
        dark = dark.load()
        white_full = np.tile(white, (img.shape[0],1,1))
        white_full_RGB = np.stack((white_full[:,:,int(img.metadata['default bands'][2])],white_full[:,:,int(img.metadata['default bands'][1])],white_full[:,:,int(img.metadata['default bands'][0])]), axis=2)
        dark_full = np.tile(dark, (img.shape[0],1,1))

        #img_normalized = np.array(img.load())
        #img_normalized = (img.load() - dark_full) / (white_full - dark_full)
        #img_normalized = img_normalized * 2
        img_normalized = sfilter((img.load() - dark_full) / (white_full - dark_full)) + 0.1
        #img_normalized = shiftw(img_normalized)
        #img_normalized = np.array(img.load())
        img_normalized[img_normalized <= 0] = 10**-2
        #print(img_normalized.shape)
        ### Visualising spectrograms

        #intensity1 = []
        #intensity2 = []

        x_blood,y_blood,z_blood = np.where(blood==1)
        #blood_index = 5
        #orange_dot = [x_blood[blood_index], y_blood[blood_index]]

        #blue_dot = [110,180]
        #orange_dot = [220,160]
        #orange_dot = [82,267]
        #wavelength = np.linspace(400, 1000, 826)
        wavelength = np.array(img.metadata['wavelength']).astype(float)
        wavelength_filtered = wavelength

        img_RGB = np.stack((img_normalized[:,:,int(img.metadata['default bands'][2])],img_normalized[:,:,int(img.metadata['default bands'][1])],img_normalized[:,:,int(img.metadata['default bands'][0])]), axis=2)

        mol_list = ["water_hale"]
        molecules_ucl, x_ucl = read_molecules(left_cut, right_cut, mol_list, wavelength_filtered)

        molecules_cyto, x_cyto = read_molecules_cytochrome_cb(left_cut, right_cut, wavelength_filtered)

        molecules, x = read_molecules_creatis(left_cut, right_cut, x_waves=wavelength_filtered)

        wavelengths_chosen = np.isin(wavelength_filtered, wl)
        x_chosen = np.isin(x, wl)

        y_c_oxy, y_c_red, y_b_oxy, y_b_red = molecules_cyto #, y_water, y_fat = molecules

        y_hb_f, y_hbo2_f, y_coxa, y_creda, y_fat, y_water = molecules

        M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                    np.asarray(y_hb_f),
                                    np.asarray(y_coxa),
                                    np.asarray(y_creda),
                                    np.asarray(y_c_oxy),
                                    np.asarray(y_c_red),
                                    np.asarray(y_b_oxy),
                                    np.asarray(y_b_red),                                    
                                    np.asarray(y_water),
                                    np.asarray(y_fat))))

        M = M[x_chosen,:]
        img_normalized = img_normalized[:,:,wavelengths_chosen]

        config.molecule_count = M.shape[1]
        config.max_b = 4
        config.max_a = 100
        coarseness = 1
        hbt_opti = np.zeros((img.shape[0], img.shape[1]))
        hbdiff_opti = np.zeros((img.shape[0], img.shape[1]))
        error_opti = np.zeros((img.shape[0], img.shape[1]))
        diffCCO_opti = np.zeros((img.shape[0], img.shape[1]))

        #reference_index_blood = random.randint(0, len(x_blood) - 1)
        #reference_point = [x_blood[reference_index_blood], y_blood[reference_index_blood]]
        reference_point = torch.load("dataset/helicoid/" + patient + "/reference_point.pt")
        #coefs, scattering_params, errors, a_t1, b_t1, img_ref = helicoid_optimisation_scattering_wlselection(img_normalized, reference_point, M, x[x_chosen], 8, patient, x_chosen)
        
        error_wavelength_selection, params_found, params_found_GT = helicoid_optimisation_scattering_wlselection(img_normalized, reference_point, M, x[x_chosen], coarseness, patient, x_chosen, label_map=gt, only_labeled=only_labeled, use_parallel=use_parallel_lsq)
        error_wavelength_selection_patients.append(error_wavelength_selection)
        params_found_patients.append(params_found)
        params_found_GT_patients[patient_id] = params_found_GT
        
    return np.mean(error_wavelength_selection_patients), params_found_patients, params_found_GT_patients 

def helicoid_run_patient():
    hdr_path = path+"/{}/raw.hdr".format("012-01")
    img = open_image(hdr_path)
    wavelength = np.array(img.metadata['wavelength']).astype(float)
    molecules, x = read_molecules_creatis(left_cut_helicoid, right_cut_helicoid, x_waves=wavelength)
    params_found_GT = None
    
    current_wavelengths = x
    
    found_set = []
    found_err = []
    found_params = []
    
    for i in range(len(x) - 1):
        metrics = np.zeros(len(current_wavelengths))
        for i in range(len(current_wavelengths)):
            x_reduced = np.delete(x, i)
            error, params_found, params_found_GT = helicoid_run_with_wl(x_reduced)
            metrics[i] = error

        index_to_remove = np.argmin(metrics)
        #index_to_remove = np.argmin(metrics)
        current_wavelengths = np.delete(current_wavelengths, index_to_remove)
        found_set.append(current_wavelengths)
        found_err.append(error)
        found_params.append(params_found)
                   
    return found_set, found_err, found_params 
    

def compute_error_for_wavelength_removal(args):
    x, only_labeled, i = args
    x_reduced = np.delete(x, i)
    error, params_found, _ = helicoid_run_with_wl(x_reduced, only_labeled=only_labeled, coarseness=4, use_parallel_lsq=True)
    return error, params_found, i

# def remove_wavelengths_around_index(current_wavelengths, index, n):
#     start_index = max(index - n, 0)
#     end_index = min(index + n + 1, len(current_wavelengths))
#     assert(len(range(start_index, end_index)) == 2 * n + 1)
#     return np.delete(current_wavelengths, range(start_index, end_index))


def helicoid_run_patient_parallel(only_labeled=False):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hdr_path = f"{path}/012-01/raw.hdr"
    img = open_image(hdr_path)
    wavelength = np.array(img.metadata['wavelength']).astype(float)
    _, x = read_molecules_creatis(left_cut_helicoid, right_cut_helicoid, x_waves=wavelength)
    params_found_GT = None
    
    current_wavelengths = x[::2]
    found_set = []
    found_err = []
    found_params = []
    found_time = []
    
    #for iteration in tqdm(range(len(x) - 1)):
    while(len(current_wavelengths) > 16):
        print("Current wavelengths" + str(current_wavelengths))
        print("Current amount of wavelengths: " + str(len(current_wavelengths)))
        args = [(current_wavelengths, only_labeled, i) for i in range(len(current_wavelengths))]
        
        start = time.time()
        with ProcessPoolExecutor(3) as executor:
            results = list(executor.map(compute_error_for_wavelength_removal, args))
        end = time.time()
        found_time.append(end-start)
        errors, params, indices = zip(*results)
        metrics = np.array(errors)
        print(len(params))
        params = params[-1] #pick worst one
        
        #index_to_remove = np.argmax(metrics)
        
        #indices_to_remove = np.argsort(metrics)[0:np.ceil(0.05*len(current_wavelengths)).astype(int)]
        indices_to_remove = np.argsort(metrics)[0]
        chosen_params_patients = []
        
        for patient in range(len(params)):
            #chosen_params = np.array(params)[indices_to_remove]
            chosen_params = np.array(params[patient])
            chosen_params_patients.append(chosen_params)
            print(chosen_params.shape)
            #print(errors)
            #print(indices)

            #current_wavelengths = np.delete(current_wavelengths, indices[index_to_remove])            
            plt.figure()
            #plt.imshow(chosen_params[0][:,:,0] + chosen_params[0][:,:,1])
            plt.imshow(chosen_params[:,:,0] + chosen_params[:,:,1],cmap="Reds")
            plt.title("HbT - Patient " + str(patient))
            plt.show()
            plt.figure()
            #plt.imshow(chosen_params[0][:,:,2] - chosen_params[0][:,:,3])
            plt.imshow(chosen_params[:,:,2] - chosen_params[:,:,3])
            plt.title("diffCCO - Patient " + str(patient))
            plt.show()
            
        print("Deleting wavelengths: " + str(current_wavelengths[indices_to_remove]))
        print("Computed for " + str(round(end-start, 2)) + " seconds")
        current_wavelengths = np.delete(current_wavelengths, indices_to_remove)
        found_set.append(current_wavelengths.copy())
        found_err.append(metrics[indices_to_remove])
        found_params.append(chosen_params_patients)
        
        filename = f"helicoid_wl_opti_{current_date}.pickle"
        with open(filename, "wb") as file:
            pickle.dump(found_set, file)
            pickle.dump(found_err, file)
            pickle.dump(found_params, file)
            pickle.dump(found_time, file)
               
    return found_set, found_err, found_params, found_time
    
    
def compute_mse(i, *args):
    #print(i)
    i = int(i[0])
    wavelength_index_combinations, M, dataset_spectr, concentrations_all_wavelengths, possible_combinations = args[0], args[1], args[2], args[3], args[4]
    
    if i % int(possible_combinations/9) == 0:
        print("1/10 iterations complete")
    
    indices = wavelength_index_combinations[i]
    concentrations = (pinv(M[indices,:]) @ dataset_spectr[:,indices].T).T
    return np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))

def compute_mse_clip(indices, *args):
    M, dataset_spectr, concentrations_all_wavelengths, opt_spectral_fit, absorption_ortho = args[0], args[1], args[2], args[3], args[4]
    #print(M.shape)
    #print(indices)
    valid_indices = np.unique(np.round(indices).astype(int))
    #print(valid_indices)
    #mse = np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))
    #print(np.mean(concentrations_all_wavelengths**2, axis=0))
    if absorption_ortho:
        mse = np.linalg.cond(M[valid_indices,:])
    else:
        concentrations = (np.linalg.pinv(M[valid_indices,:]) @ dataset_spectr[:, valid_indices].T).T
        if opt_spectral_fit:
            mse = np.sqrt(np.mean(np.square((M @ concentrations.T) - dataset_spectr[:, :].T)))
        else:
            mse = np.mean(np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2, axis=0) / np.mean(concentrations_all_wavelengths**2, axis=0)))
    return mse

def map_b_to_a(a, b):
    return np.argmin(np.abs(a - b[:, np.newaxis]), axis=1)

def compute_mse_scipy(wavelengths_to_probe, *args):
    wavelengths, M, dataset_spectr, concentrations_all_wavelengths = args[0], args[1], args[2], args[3]
    indices = map_b_to_a(wavelengths, np.cumsum(wavelengths_to_probe))
    concentrations = (pinv(M[indices,:]) @ dataset_spectr[:,indices].T).T
    return np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))

def wavelength_opti_greedy(desired_num_wavelengths, left_cut, right_cut, coarseness_wl=5, coarseness_data=100, use_light=False, opt_spectral_fit=False, absorption_ortho=False):
    if use_light:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra_light(left_cut, right_cut, coarseness_wl=coarseness_wl, coarseness_data=coarseness_data)
    else:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra(left_cut, right_cut)
        
    M_allwl = np.copy(M)
    dataset_spectr_allwl = np.copy(dataset_spectr)
    concentrations_all_wavelengths = (pinv(M)@dataset_spectr.T).T
    
    remaining_indices = np.arange(M.shape[0])
    print("Optimizing greedy")
    with tqdm(total = (M.shape[0] - desired_num_wavelengths)) as pbar:
        while M.shape[0] > desired_num_wavelengths:
            metrics = []
            for i in (range(M.shape[0])):
                M_reduced = np.delete(M, i, axis=0)
                dataset_spectr_reduced = np.delete(dataset_spectr, i, axis=1)
                #metric = np.sqrt(np.mean((((pinv(M_reduced) @ dataset_spectr_reduced.T).T) - concentrations_all_wavelengths)**2))
                concentrations = (pinv(M_reduced) @ dataset_spectr_reduced.T).T
                
                if absorption_ortho:
                    metric = np.linalg.cond(M_reduced)
                else:
                    if opt_spectral_fit:
                        metric = np.sqrt(np.mean(np.square((M_allwl @ concentrations.T) - dataset_spectr_allwl[:, :].T)))
                    else:
                        metric = np.mean(np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2, axis=0) / np.mean(concentrations_all_wavelengths**2, axis=0)))
                
                metrics.append(metric)

            index_to_remove = np.argmin(metrics)
            #index_to_remove = np.argmin(metrics)
            M = np.delete(M, index_to_remove, axis=0)
            dataset_spectr = np.delete(dataset_spectr, index_to_remove, axis=1)
            remaining_indices = np.delete(remaining_indices, index_to_remove)
            pbar.update(1)
               
    print(remaining_indices)
    concentrations = (pinv(M) @ dataset_spectr.T).T
    if opt_spectral_fit:
        final_mse = np.sqrt(np.mean(np.square((M_allwl @ concentrations.T) - dataset_spectr_allwl[:, :].T)))
    else:
        final_mse = np.mean(np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2, axis=0) / np.mean(concentrations_all_wavelengths**2, axis=0)))
    return remaining_indices, wavelengths[remaining_indices], concentrations_all_wavelengths, final_mse
    # constraints = {'type': 'ineq', 'fun': constraint_function, 'args': (wavelengths)}
    # bounds = bounds=(np.min(wavelengths), np.max(wavelengths)) + ((0, np.max(wavelengths) - np.min(wavelengths)),)*(num_wavelengths-1)
    # start = time.time()
    # scipy.optimize.differential_evolution(compute_mse_scipy, workers=16, bounds=bounds, args=(wavelengths, M, dataset_spectr, concentrations_all_wavelengths), constraints=constraints)

# def calculate_mse(params):
#     indices, M, dataset_spectr, concentrations_all_wavelengths = params
#     concentrations = (pinv(M[indices,:]) @ dataset_spectr[:,indices].T).T
#     mse = np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))
#     return mse

# def wavelength_opti_chunked(num_wavelengths, left_cut, right_cut):
#     wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra(left_cut, right_cut)
#     concentrations_all_wavelengths = (pinv(M)@dataset_spectr.T).T
#     #wavelength_index_combinations = np.array(list(itertools.combinations(range(len(wavelengths)), num_wavelengths)))
#     #possible_combinations = len(wavelength_index_combinations)

#     def combinations_chunk_generator(total, chunk_size, num_wavelengths):
#         iterator = itertools.combinations(range(total), num_wavelengths)
#         while True:
#             chunk = list(itertools.islice(iterator, chunk_size))
#             if not chunk:
#                 return
#             yield chunk
            
#     num_samples = np.math.comb(len(wavelengths), num_wavelengths)
#     chunk_size = 1000  # Adjust based on memory and performance requirements
#     num_processes = 64  # Adjust based on your machine's capabilities

#     with Pool(num_processes) as pool:
#         mse_results = []

#         for chunk in tqdm(combinations_chunk_generator(len(wavelengths), chunk_size, num_wavelengths)):
#             results = pool.map(calculate_mse, [(indices, M, dataset_spectr, concentrations_all_wavelengths) for indices in chunk])
#             mse_results.extend(results)

#         pool.close()
#         pool.join()

#     # Output the length of mse_results to verify
#     len(mse_results)
#     return mse_results
    # num_samples = wavelength_index_combinations.shape[0]
    # wavelength_mse = np.zeros(possible_combinations) + np.nan
    #wavelength_index_combinations_shuffled = np.random.permutation(wavelength_index_combinations)

    #for parallel, use scipy bruteforce, you need it anyway for differential evolution    
    # for i in tqdm(range(num_samples)):
    #     indices = wavelength_index_combinations[i]
    #     concentrations = (pinv(M[indices,:]) @ dataset_spectr[:,indices].T).T
    #     wavelength_mse[i] = np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))

# def combination_generator(n, r, start, end):
#     return itertools.islice(itertools.combinations(range(n), r), start, end)

# def optimize_combinations(start, end, wavelengths, M, dataset_spectr, concentrations_all_wavelengths, num_wavelengths):
#     best_combination = None
#     best_mse = float('inf')

#     with tqdm(total=end-start) as pbar:
#         for indices in (combination_generator(len(wavelengths), num_wavelengths, start, end)):
#             concentrations = (pinv(M[np.array(indices), :]) @ dataset_spectr[:, np.array(indices)].T).T
#             mse = (np.mean((concentrations - concentrations_all_wavelengths)**2))

#             if mse < best_mse:
#                 best_mse = mse
#                 best_combination = indices
#             pbar.update(1)

#     return best_combination, best_mse

# def wavelength_opti_brute_parallel(num_wavelengths, left_cut, right_cut):
#     wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra(left_cut, right_cut)
#     dataset_spectr = dataset_spectr[0:1000,:]
#     concentrations_all_wavelengths = (pinv(M)@dataset_spectr.T).T
#     total_combinations = math.comb(len(wavelengths), num_wavelengths)
#     num_processes = 32
#     range_size = total_combinations // num_processes
#     ranges = [(i * range_size, (i + 1) * range_size) for i in range(num_processes)]

#     # Prepare arguments for each process
#     process_args = [(start, end, wavelengths, M, dataset_spectr, concentrations_all_wavelengths, num_wavelengths) for start, end in ranges]

#     with multiprocessing.Pool() as pool:
#         results = pool.starmap(optimize_combinations, process_args)

#     best_result = min(results, key=lambda x: x[1])
#     print(best_result)
#     print("Best Combination:", best_result[0], "with MSE:", best_result[1])

def wavelength_opti_greedy_sfs(desired_num_wavelengths, left_cut, right_cut, start_wavelengths, use_light=False):
    if use_light:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra_light(left_cut, right_cut)
    else:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra(left_cut, right_cut)
    concentrations_all_wavelengths = (pinv(M)@dataset_spectr.T).T
    
    result_brute = wavelength_opti_brute_parallel(start_wavelengths, left_cut, right_cut, use_light=use_light)
    M_brute = M[result_brute[0], :]
    dataset_spectr_brute = dataset_spectr[:, result_brute[0]]
    remaining_indices = np.arange(M_red.shape[0])
    
    print("Optimizing greedy SFS")
    with tqdm(total = (M.shape[0] - desired_num_wavelengths)) as pbar:
        while M.shape[0] < desired_num_wavelengths:
            metrics = []
            for i in (range(M.shape[0])):
                #M_reduced = np.delete(M, i, axis=0)
                dataset_spectr_reduced = np.delete(dataset_spectr, i, axis=1)
                #metric = np.sqrt(np.mean((((pinv(M_reduced) @ dataset_spectr_reduced.T).T) - concentrations_all_wavelengths)**2))
                concentrations = (pinv(M_reduced) @ dataset_spectr_reduced.T).T
                metric = np.mean(np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2, axis=0) / np.mean(concentrations_all_wavelengths**2, axis=0)))
                metrics.append(metric)

            index_to_remove = np.argmin(metrics)
            #index_to_remove = np.argmin(metrics)
            M = np.delete(M, index_to_remove, axis=0)
            dataset_spectr = np.delete(dataset_spectr, index_to_remove, axis=1)
            remaining_indices = np.delete(remaining_indices, index_to_remove)
            pbar.update(1)
               
    print(remaining_indices)
    concentrations = (pinv(M) @ dataset_spectr.T).T
    final_mse = np.mean(np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2, axis=0) / np.mean(concentrations_all_wavelengths**2, axis=0)))
    return remaining_indices, wavelengths[remaining_indices], concentrations_all_wavelengths, final_mse
  
def get_adjusted_indices(i, n_adjacent, total_length):
    if i <= n_adjacent:
        start, end = 0, min(2 * n_adjacent + 1, total_length)
    elif i >= total_length - n_adjacent - 1:
        start, end = max(0, total_length - 2 * n_adjacent - 1), total_length
    else:
        start, end = i - n_adjacent, i + n_adjacent + 1
    return list(range(start, end))

def expand_indices_with_adjacent(array, n_adjacent, total_length):
    all_indices = [get_adjusted_indices(i, n_adjacent, total_length) for i in array]
    combined_indices = []
    last_val = None
    for sublist in all_indices:
        for index in sublist:
            if index != last_val:
                combined_indices.append(index)
                last_val = index
            
    return combined_indices

def combination_generator(n, r, start, end, n_adjacent):
    return itertools.islice(itertools.combinations(range(0 + n_adjacent, n - n_adjacent), r), start, end)

def optimize_combinations(start, end, wavelengths, M, dataset_spectr, concentrations_all_wavelengths, num_wavelengths, n_adjacent, opt_spectral_fit, absorption_ortho, opt_projection):
    best_combination = None
    best_mse = float('inf')

    #with tqdm(total=end-start) as pbar:
    for indices in (combination_generator(len(wavelengths), num_wavelengths, start, end, n_adjacent)):
        indices = expand_indices_with_adjacent(indices, n_adjacent, len(wavelengths))
        #mse = (np.mean((concentrations - concentrations_all_wavelengths)**2))
        if opt_projection:
            concentrations = (pinv(M[np.array(indices), :]) @ dataset_spectr[:, np.array(indices)].T).T
            mse = np.sqrt(np.mean(np.square((M[np.array(indices), :] @ concentrations.T) - dataset_spectr[:, np.array(indices)].T)))
        else:
            if absorption_ortho:
                mse = np.linalg.cond(M[np.array(indices), :])
            else:
                concentrations = (pinv(M[np.array(indices), :]) @ dataset_spectr[:, np.array(indices)].T).T
                if opt_spectral_fit:
                    mse = np.sqrt(np.mean(np.square((M @ concentrations.T) - dataset_spectr[:, :].T)))
                else:
                    mse = np.mean(np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2, axis=0) / np.mean(concentrations_all_wavelengths**2, axis=0)))

        if mse < best_mse:
            best_mse = mse
            best_combination = indices
    #        pbar.update(1)
    print("Processed with range: " + str(start) + " - " + str(end) + " with best MSE: " + str(best_mse) + " and combination: " + str(best_combination) + " has finished")
    return best_combination, best_mse

def wavelength_opti_brute_parallel(num_wavelengths, left_cut, right_cut, n_adjacent=0, coarseness_wl=5, coarseness_data=100, use_light=False, opt_spectral_fit=False, absorption_ortho=False, opt_projection=False):
    if use_light:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra_light(left_cut, right_cut, coarseness_wl=coarseness_wl, coarseness_data=coarseness_data)
    else:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra(left_cut, right_cut)
    #dataset_spectr = dataset_spectr[:,:]
    concentrations_all_wavelengths = (pinv(M)@dataset_spectr.T).T
    total_combinations = math.comb(len(wavelengths) - 2*n_adjacent, num_wavelengths)
    num_processes = 128
    range_size = total_combinations // num_processes
    ranges = [(i * range_size, (i + 1) * range_size) for i in range(num_processes)]

    # Prepare arguments for each process
    process_args = [(start, end, wavelengths, M, dataset_spectr, concentrations_all_wavelengths, num_wavelengths, n_adjacent, opt_spectral_fit, absorption_ortho, opt_projection) for start, end in ranges]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(optimize_combinations, process_args)

    best_result = min(results, key=lambda x: x[1])
    print(best_result)
    print("Best Combination:", best_result[0], "with MSE:", best_result[1])
    return best_result

def wavelength_opti(num_wavelengths, left_cut, right_cut, coarseness_wl=5, coarseness_data=100, optimization_method="brute", use_light=False, opt_spectral_fit=False, absorption_ortho=False):
    if use_light:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra_light(left_cut, right_cut, coarseness_wl=coarseness_wl, coarseness_data=coarseness_data)
    else:
        wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra(left_cut, right_cut)
        
    concentrations_all_wavelengths = (pinv(M)@dataset_spectr.T).T
    #wavelength_index_combinations = np.array(list(itertools.combinations(range(len(wavelengths)), num_wavelengths)))
    #possible_combinations = len(wavelength_index_combinations)
    #print("Computed combinations - starting compute")
    # num_samples = wavelength_index_combinations.shape[0]
    # wavelength_mse = np.zeros(possible_combinations) + np.nan
    #wavelength_index_combinations_shuffled = np.random.permutation(wavelength_index_combinations)

    #for parallel, use scipy bruteforce, you need it anyway for differential evolution    
    # for i in tqdm(range(num_samples)):
        # indices = wavelength_index_combinations[i]
        # concentrations = (pinv(M[indices,:]) @ dataset_spectr[:,indices].T).T
        # wavelength_mse[i] = np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))

    # def calculate_mse(indices, M, dataset_spectr, concentrations_all_wavelengths):
    #     concentrations = (pinv(M[indices[0],:]) @ dataset_spectr[:,indices[0]].T).T
    #     return np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))

    # def objective_function(i):
    #     indices = wavelength_index_combinations[i]
    #     return calculate_mse(indices, M, dataset_spectr, concentrations_all_wavelengths)

    # result = brute(objective_function, ((slice(0, possible_combinations,1),)), full_output=True, finish=None)
    start = time.time()
    if optimization_method == "brute":
        result = brute(compute_mse, ((slice(0, possible_combinations,1),)), args=(wavelength_index_combinations, M, dataset_spectr, concentrations_all_wavelengths, possible_combinations), full_output=True, finish=None, workers=64)
        print(result)
        np.savez_compressed(path_wlopti + "wl_brute_opti_result_" + str(num_wavelengths) + "_" + str(left_cut) + "_" + str(right_cut), 
                            optimal_wavelengths = np.array(result[0]), 
                            wls_error = np.array(result[1]), 
#                            indices_flat = result[2], 
#                            error_indices = result[3],
                            wavelengths = wavelengths,
                            chromophores = np.array([y_hbo2_f, y_hb_f, y_coxa - y_creda]),
                            M = M,
                            concentrations_all_wavelengths = concentrations_all_wavelengths)
        optimal_indices = result[0].astype(int)
        end = time.time()
        print("Time elapsed: " + str(end-start))
        return wavelength_index_combinations[optimal_indices], wavelengths, result, concentrations_all_wavelengths
    elif optimization_method == "differential_evolution":
        #result = differential_evolution(compute_mse, bounds=((0, possible_combinations-1),), args=(wavelength_index_combinations, M, dataset_spectr, concentrations_all_wavelengths, possible_combinations), workers=64)
        #return wavelength_index_combinations[int(result.x)], wavelengths[wavelength_index_combinations[int(result.x)]], result, concentrations_all_wavelengths
        bounds = [(0, len(wavelengths) - 1)] * num_wavelengths
        #result = differential_evolution(compute_mse_clip, bounds, args=(M, dataset_spectr, concentrations_all_wavelengths), maxiter=10000, popsize=30, workers=32)
        result = differential_evolution(compute_mse_clip, bounds, args=(M, dataset_spectr, concentrations_all_wavelengths, opt_spectral_fit, absorption_ortho), workers=32)
        return result
    elif optimization_method == "dual_annealing":
        result = dual_annealing(compute_mse, bounds=[(0, len(wavelengths) - 1)] * num_wavelengths, args=(M, dataset_spectr, concentrations_all_wavelengths))
        return wavelength_index_combinations[int(result.x)], wavelengths[wavelength_index_combinations[int(result.x)]], result, concentrations_all_wavelengths
    else:
        raise("Optimization method not supported")
    
    # #print(result)
    # optimal_indices = result[0].astype(int)
    # #print(optimal_indices)
    # optimal_wavelength_errors = result[1]
    # #print(optimal_wavelength_errors)
    # wavelength_mse[optimal_indices] = optimal_wavelength_errors
    # return wavelength_index_combinations[optimal_indices], result, concentrations_all_wavelengths

    # with futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     results = list(tqdm(executor.map(partial(calculate_mse, wavelength_index_combinations=wavelength_index_combinations, M=M, dataset_spectr=dataset_spectr, concentrations_all_wavelengths=concentrations_all_wavelengths), range(num_samples)), total=num_samples))

    # for i, result in enumerate(results):
    #     wavelength_mse[i] = result
        
    # np.savez_compressed(path_wlopti + "wavelength_mse_wavelengths_" + str(num_wavelengths) + "_" + str(left_cut) + "_" + str(right_cut), wavelength_mse)
    # sorted_indices = np.argsort(wavelength_mse)
    # optimal_wavelength_indices = np.unravel_index(sorted_indices, wavelength_mse.shape)[0]
    # optimal_wavelength_errors = wavelength_mse.flatten()[sorted_indices]
    # #best_index = wavelength_index_combinations[optimal_wavelength_indices[0]]
    
    # return wavelength_index_combinations[optimal_wavelength_indices], optimal_wavelength_errors, concentrations_all_wavelengths

# def update_array(data_tuple, i):
#     wavelength_index_combinations, dataset_spectr, M, concentrations_all_wavelengths = data_tuple
#     indices = wavelength_index_combinations[i]
#     concentrations = (pinv(M[indices,:]) @ dataset_spectr[:,indices].T).T
#     return np.sqrt(np.mean((concentrations - concentrations_all_wavelengths)**2))
        
# def brute_force_wavelength_opti(num_wavelengths, left_cut, right_cut):
#     wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_piglet_spectra(left_cut, right_cut)
#     concentrations_all_wavelengths = (pinv(M)@dataset_spectr.T).T
#     wavelength_index_combinations = np.array(list(itertools.combinations(range(len(wavelengths)), num_wavelengths)))
#     possible_combinations = len(wavelength_index_combinations)
#     num_samples = wavelength_index_combinations.shape[0]
#     wavelength_mse = np.zeros(possible_combinations) + np.nan
    
#     data_tuple = (wavelength_index_combinations, dataset_spectr, M, concentrations_all_wavelengths)
#     partial_update_array = partial(update_array, data_tuple)
    
#     with Pool(processes=4) as pool:
#         results = process_map(partial_update_array, range(num_samples), total=num_samples, chunksize=1)
        
#     for i, result in enumerate(results):
#         wavelength_mse[i] = result

#     np.savez_compressed(path_wlopti + "wavelength_mse_wavelengths_" + str(num_wavelengths) + "_" + str(left_cut) + "_" + str(right_cut), wavelength_mse)
#     sorted_indices = np.argsort(wavelength_mse)
#     optimal_wavelength_indices = np.unravel_index(sorted_indices, wavelength_mse.shape)[0]
#     optimal_wavelength_errors = wavelength_mse.flatten()[sorted_indices]
    
    # return wavelength_index_combinations[optimal_wavelength_indices], concentrations_all_wavelengths

def plot_optimal_wavelengths(optimal_wavelength_indices, concentrations_all_wavelengths, wavelengths, M, dataset_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda, left_cut, right_cut):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    best_index = optimal_wavelength_indices[0]
    best_concentration = (pinv(M[best_index,:]) @ dataset_spectr[:,best_index].T).T
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    for i in range(concentrations_all_wavelengths.shape[1]):
        axs[0].plot(concentrations_all_wavelengths[:,i], color=colors[i], linewidth=4, alpha=0.3)
        axs[0].plot(best_concentration[:,i], linewidth=1)
        #concentrations_all_wavelengths.shape
    axs[0].set_title("Inferred Concentrations")

    axs[1].plot(wavelengths, y_hbo2_f)
    axs[1].plot(wavelengths, y_hb_f)
    axs[1].plot(wavelengths, y_coxa - y_creda)
    axs[1].set_title("Chromophores and used wavelengths")

    for val in wavelengths[best_index]:
        axs[1].axvline(x=val, color='black', linestyle='--', alpha=0.7)
        
        
if __name__ == "__main__":
    print(wavelength_opti_brute_parallel(5, 780, 900, n_adjacent=0))
    #print(wavelength_opti_greedy(3, 780, 900))
    
    #wavelength_opti(4, 780, 900, optimization_method="brute")
    #wavelength_opti(4, 740, 900, optimization_method="brute")
    # wavelength_opti(5, 780, 900, optimization_method="brute")
    # wavelength_opti(5, 740, 900, optimization_method="brute")
    pass