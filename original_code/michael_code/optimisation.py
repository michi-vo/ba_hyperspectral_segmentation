import os
import scipy.io
import cvxpy as cp
import seaborn as sns
import numpy as np
import pickle
import time
import timeit
import scipy
from tqdm.auto import tqdm
from scipy.linalg import pinv
import matplotlib.pyplot as plt
from itertools import combinations

import data_processing.preprocessing as preprocessing
#from config import left_cut, right_cut, molecule_count, max_b, max_a, scattering_model
import config
import utils

"""
Script the runs cvxpy optimisation, to find the param differences between two spectra
Last line saves the result for later use in NN training.
"""

sns.set()

# extract file names
folder = os.listdir("dataset/miniCYRIL Piglet Data")
folder = [i for i in folder if "lwp" in i or "LWP" in i]
dic = {}
dic['lwp475'] = "LWP475_Ws_14Nov2016_1.mat"
dic['lwp478'] = "LWP474_Ws_07Nov2016.mat" #there is no piglet folder 474 so this is fine...
dic['lwp479-HI+saline'] = "LWP479_Ws_05Dec2016.mat"
dic['lwp494'] = "LWP494_Ws_27Mar2017_17  21.mat"
dic['lwp499'] = "LWP499_Ws_02_May_2017.mat"
dic['LWP480'] = "LWP480_Ws_12Dec2016.mat"
dic['LWP481'] = "LWP481_Ws_03Jan2017.mat"
dic['LWP484'] = "LWP484_Ws_23Jan2017.mat"
dic['LWP485'] = "LWP485_Ws_30Jan2017.mat"
dic['LWP489'] = "LWP489_Ws_20Feb2017.mat"
dic['LWP488'] = "LWP488_Ws_13Feb2017.mat"
dic['LWP490'] = "LWP490_Ws_27Feb2017.mat"
dic['LWP491'] = "ResultsLWP491_06Mar2017.mat" #special case
dic['LWP492'] = "LWP492_Ws_13Mar2017.mat"
dic['LWP495'] = "LWP495_Ws_03Apr2017_14  23.mat"
dic['LWP498'] = "LWP498_Ws_24Apr2017_15.mat"
dic['LWP500'] = "LWP500_Ws_15_May_201716.mat"
dic['LWP501'] = "LWP501_Ws_22_May_201715.mat"
dic['LWP502'] = "LWP502_Ws_30_May_201716.mat"
dic['LWP503'] = "LWP503_Ws_05_Jun_2017_15   8.mat"
dic['LWP504'] = "LWP504_Ws_12Jun_2017_17   1.mat"
dic['LWP507'] = "LWP507_Ws_10Jul_2017_16  55.mat"
dic['LWP509'] = "LWP509_Ws_17Jul_2017_11  21.mat"
dic['LWP511'] = "LWP511_Ws_25Jul_2017_16   2.mat"
dic['LWP512'] = "LWP512_Ws_31Jul_2017_17  16.mat"

wavelengths = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/lwp475/' + 'LWP475_Ws_14Nov2016_1.mat')['wavelengths'].astype(float)
idx = (wavelengths >= config.left_cut) & (wavelengths <= config.right_cut)
wavelengths = wavelengths[idx]
m = config.molecule_count  # number of parameters (from 2 to 6)
molecules, x = preprocessing.read_molecules(config.left_cut, config.right_cut, wavelengths)
y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _, y_cytoa_diff, _, _, _, _ = molecules
M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                            np.asarray(y_hb_f),
                            np.asarray(y_coxa - y_creda))))

img_whitecount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/refSpectrum.mat')
img_darkcount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/LWP480_DarkCount_12Dec2016.mat')

white_full = img_whitecount['refSpectrum'].astype(float)
dark_full = img_darkcount['DarkCount'].astype(float)
white_full = white_full[idx.squeeze()]
dark_full = dark_full[idx.squeeze()]

def get_spectra(pig, left_cut, right_cut):
    wavelengths = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/lwp475/' + 'LWP475_Ws_14Nov2016_1.mat')['wavelengths'].astype(float)
    idx = (wavelengths >= left_cut) & (wavelengths <= right_cut)
    wavelengths = wavelengths[idx]
    m = config.molecule_count  # number of parameters (from 2 to 6)
    molecules, x = preprocessing.read_molecules(left_cut, right_cut, wavelengths)
    y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _, y_cytoa_diff, _, _, _, _ = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa - y_creda))))

    img_whitecount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/refSpectrum.mat')
    img_darkcount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/LWP480_DarkCount_12Dec2016.mat')

    white_full = img_whitecount['refSpectrum'].astype(float)
    dark_full = img_darkcount['DarkCount'].astype(float)
    white_full = white_full[idx.squeeze()]
    dark_full = dark_full[idx.squeeze()]
    spectr, _ = preprocess_piglet(pig, left_cut=left_cut, right_cut=right_cut)
    spectr = (spectr[:,:1000])
    ref_spectr = (spectr[:, 0] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
    comp_spectr = np.array([(spectr[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0]) for i in range(0,1000)])
    comp_spectr = np.log(1 / (comp_spectr / ref_spectr))
    return wavelengths, M, comp_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda

def get_spectra_all_chromophores(pig, left_cut, right_cut):
    wavelengths = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/lwp475/' + 'LWP475_Ws_14Nov2016_1.mat')['wavelengths'].astype(float)
    idx = (wavelengths >= left_cut) & (wavelengths <= right_cut)
    wavelengths = wavelengths[idx]
    m = config.molecule_count  # number of parameters (from 2 to 6)
    molecules, x = preprocessing.read_molecules(left_cut, right_cut, wavelengths)
    y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water, y_fat, _, y_cytoc_oxy, y_cytoc_red, y_cytob_oxy, y_cytob_red = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa - y_creda))))

    img_whitecount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/refSpectrum.mat')
    img_darkcount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/LWP480_DarkCount_12Dec2016.mat')

    white_full = img_whitecount['refSpectrum'].astype(float)
    dark_full = img_darkcount['DarkCount'].astype(float)
    white_full = white_full[idx.squeeze()]
    dark_full = dark_full[idx.squeeze()]
    spectr, _ = preprocess_piglet(pig, left_cut=left_cut, right_cut=right_cut)
    spectr = (spectr[:,:1000])
    ref_spectr = (spectr[:, 0] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
    comp_spectr = np.array([(spectr[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0]) for i in range(0,1000)])
    comp_spectr = np.log(1 / (comp_spectr / ref_spectr))
    return wavelengths, M, comp_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water, y_fat, y_cytoc_oxy, y_cytoc_red, y_cytob_oxy, y_cytob_red

def get_spectra_all_chromophores(left_cut, right_cut):
    wavelengths = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/lwp475/' + 'LWP475_Ws_14Nov2016_1.mat')['wavelengths'].astype(float)
    idx = (wavelengths >= left_cut) & (wavelengths <= right_cut)
    wavelengths = wavelengths[idx]
    m = config.molecule_count  # number of parameters (from 2 to 6)
    molecules, x = preprocessing.read_molecules(left_cut, right_cut, wavelengths)
    y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water, y_fat, _, y_cytoc_oxy, y_cytoc_red, y_cytob_oxy, y_cytob_red = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa - y_creda))))

    img_whitecount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/refSpectrum.mat')
    img_darkcount = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/LWP480/LWP480_DarkCount_12Dec2016.mat')

    white_full = img_whitecount['refSpectrum'].astype(float)
    dark_full = img_darkcount['DarkCount'].astype(float)
    white_full = white_full[idx.squeeze()]
    dark_full = dark_full[idx.squeeze()]
    
    #spectr, _ = preprocess_piglet(pig, left_cut=left_cut, right_cut=right_cut)
    #spectr = (spectr[:,:1000])
    piglet_train_set = list(filter(lambda x: x not in ['LWP503','LWP504','LWP507', 'LWP509','LWP511','LWP512'], list(dic.keys())))
    print(piglet_train_set)
    dataset_spectr = np.array([])
    for piglet in piglet_train_set:
        wavelengths, M, comp_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda = get_spectra(piglet, left_cut, right_cut)
        dataset_spectr = np.concatenate((dataset_spectr, comp_spectr[:,:1000]), axis=0) if dataset_spectr.size else comp_spectr
    
    ref_spectr = (dataset_spectr.T[:, 0] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
    comp_spectr = np.array([(dataset_spectr.T[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0]) for i in range(0,dataset_spectr.shape[0])])
    comp_spectr = np.log(1 / (comp_spectr / ref_spectr))
    return x, M, comp_spectr, y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water, y_fat, y_cytoc_oxy, y_cytoc_red, y_cytob_oxy, y_cytob_red


def find_chromophores(chromophores, chromophore_names, projection_metric, chromophore_metric, spectr):
    projection_result = []
    for k in tqdm(range(1, len(chromophores) + 1)):
        results = []
        results_names = []
        for combo_indices in combinations(range(len(chromophores)), k):
            combo = [chromophores[i] for i in combo_indices] 
            results.append(np.transpose(np.vstack(combo)))
            #results_names.append('+'.join([chromophore_names[i] for i in combo_indices]))
            results_names.append(([chromophore_names[i] for i in combo_indices]))

        for i, M in enumerate(results):
            #wavelengths_downsampled = wavelengths[::5]
            #M_downsampled = M[::5,:]
            #comp_spectr_downsampled = comp_spectr[::7,::5]
            #plt.figure()
            #print(results_names[i])
            #M_red, remaining_indices = reduce_wavelengths(M_downsampled, comp_spectr_downsampled.T[:, :], num_of_final_wavelengths, projection_metric)
            #plt.plot(wavelengths_downsampled, M_downsampled)
            #plt.vlines(wavelengths_downsampled[remaining_indices], np.min(M_downsampled), np.max(M_downsampled), color="black", linestyle="dashed", alpha=0.5)
            #print(wavelengths_downsampled[remaining_indices])
            projection_error = projection_metric(M,spectr.T)
            #print(projection_error)
            #projection_result.append((k, results_names[i], M, projection_error, 1 / (np.prod(np.linalg.svd(M, compute_uv=False))/k)))    
            projection_result.append((k, results_names[i], M, projection_error, chromophore_metric(M, spectr.T)))    

            
    return projection_result

def preprocess_piglet(pig, left_cut=config.left_cut, right_cut=config.right_cut):
    date = dic[pig]
    img = scipy.io.loadmat('dataset/miniCYRIL Piglet Data/' + pig + '/' + date)
    if pig == 'LWP491':
        #print(img['Rawdata']['spectralDataAll'][0,0][0][0].astype(float).shape)
        wavelengths_found = img['Rawdata']['wavelengths'][0,0][0][0].astype(float)
        spectr = img['Rawdata']['spectralDataAll'][0,0][0][0].astype(float)
        #print(spectr.shape)
    else:
        wavelengths_found = img['wavelengths'].astype(float)
        spectr = img['spectralDataAll'].astype(float)    
        #print(spectr.shape)        
    
    # try:
    #     paper_concentrations = img['AllConcentration'].astype(float)
    # except:
    #     paper_concentrations = []
    if pig == "LWP491":
        paper_concentrations = img['AllConcentration'][0,0].astype(float)
        #print(img['AllConcentration'][0,0].shape)
    else:
        paper_concentrations = img['AllConcentration'].astype(float)

    #print(dark_full)

    idx = (wavelengths_found >= left_cut) & (wavelengths_found <= right_cut)
    wavelengths_found = wavelengths_found[idx]
    
    #assert(np.all(wavelengths_found == wavelengths))
    
    spectr = spectr[idx.squeeze()]
    spectr[spectr <= 0] = 0.0001

    #print(white_full.shape, dark_full.shape, wavelengths.shape, spectr.shape) 
    return spectr, paper_concentrations

def plot_concentrations(ax, concentrations_no_scatter, coef_list_v):
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    coef = ["HbO2", "Hbb","diffCCO"]
    for i in range(config.molecule_count):
        ax.plot(concentrations_no_scatter[i, :], '--',label=coef[i] + ' (Standard Model)', color=colors[i], linewidth=2, alpha=0.8)
        ax.plot(coef_list_v[:, i], label=coef[i] + ' (Scattering Model)', color=colors[i], linewidth=2)
    
    ax.legend()
    ax.set_xlabel("Timepoint $t_i$", fontsize=14)
    ax.set_ylabel("$\Delta$Concentration [$mM/cm$]", fontsize=14)
    ax.set_title("Inferred Concentrations", fontsize=14)

def S(a_ti, b_ti, b_t1, a_t1):
    result = ((((wavelengths/500)**(-b_ti)) * a_ti) - (((wavelengths/500)**(-b_t1)) * a_t1)) / (1-0.9)
    return result

def f(X,*arg):
    #m = config.molecule_count
    b = arg[0]
    b_t1, a_t1 = arg[1], arg[2]
    M = arg[3]
    m = M.shape[1]
    delta_c_i = X[:m]
    a_ti = X[m:(m+1)]
    b_ti = X[(m+1):(m+2)]  
    return (M @ delta_c_i) + S(a_ti, b_ti, b_t1, a_t1) - b

def optimisation_ti(params_t1,*args):
    #print(params_t1)
    #print(len(args))
    a_t1, b_t1 = params_t1[0], params_t1[1]
    b = args[0]
    M = args[1]
    left_bound = np.append(np.ones(config.molecule_count)*(-np.inf), [-np.inf, 0])
    right_bound = np.append(np.ones(config.molecule_count)*np.inf, [np.inf, config.max_b])
    
    current_x = np.zeros(config.molecule_count+2)
    current_x[-2] = a_t1
    current_x[-1] = b_t1
    
    coef_list = []
    scattering_params_list = []
    errors_scatter = []
    
    for i in tqdm(range(0, 1000)):
        result = scipy.optimize.least_squares(f, current_x, args=(b[i], b_t1, a_t1, M), bounds=(left_bound, right_bound))
        current_x = result.x
        coef_list.append(result.x[:config.molecule_count])
        scattering_params_list.append(result.x[config.molecule_count:])
        errors_scatter.append(np.sqrt(2*result.cost))
        
    error = sum(errors_scatter) / len(errors_scatter)
    return error, coef_list, scattering_params_list

def optimisation_ti_error(params_t1, *args):
    error, _, _ = optimisation_ti(params_t1, *args)
    return error

def optimisation_scattering(spectr1, spectr2, wavelengths,pig,date,paper_concentrations):
    directory = './dataset/piglet_scattering/'+pig+'_'+str(date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    m = config.molecule_count  # number of parameters (from 2 to 6)
    molecules, x = preprocessing.read_molecules(config.left_cut, config.right_cut, wavelengths)
    y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _, y_cytoa_diff, _, _, _, _ = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa - y_creda))))
    M_pinv = pinv(M)
    # print(spectr2.shape)
    b = spectr2 / spectr1
    b = np.log(1 / np.asarray(b))
    
    start_time = time.time()
    #result = scipy.optimize.brute(optimisation_ti_error, (slice(0,config.max_a, config.max_a/2), slice(0,config.max_b, config.max_b/2)), args=(b,M), finish=None, full_output=True, workers=4)
    #result = scipy.optimize.brute(optimisation_ti_error, (slice(0,config.max_a, config.max_a/5), slice(0,config.max_b, config.max_b/150)), args=(b,M), finish=None, full_output=True, workers=64)    
    result = scipy.optimize.brute(optimisation_ti_error, (slice(0,config.max_a, config.max_a/5), slice(0,config.max_b, config.max_b/200)), args=(b,M), finish=None, full_output=True, workers=64)    
    end_time = time.time()
    
    time_taken = end_time - start_time
    #print(result)
    x_min = result[0]
    error_min = result[1]
    grid = result[2]
    error_grid = result[3]
    
    grid_x, grid_y = grid
    plt.figure()
    plt.pcolormesh(grid_x, grid_y, error_grid)
    clb = plt.colorbar()
    clb.set_label("Error")
    plt.xlabel("a(t_1)")
    plt.ylabel("b(t_1)")
    plt.savefig(directory+'/'+'error')
    
    error, coef_list, scattering_params_list = optimisation_ti(x_min,*(b,M))
    
    attenuation = b #variable b is the same as attenuation, we rename to avoid confusion
    
    n = 0
    attenuation_error_noscatter = 0
    attenuation_error_scatter = 0
    # spectrum_error_noscatter = 0
    # spectrum_error_scatter = 0
    
    attenuation_error_noscatter_780 = 0
    attenuation_error_scatter_780 = 0
    # spectrum_error_noscatter_780 = 0
    # spectrum_error_scatter_780 = 0
    
    print(x[x < 780])
    
    for i in range(0,1000):
        fig, ax = plt.subplots(1,2,figsize=(16,6))
        ax[0].plot(wavelengths, attenuation[i], label='GT Attenuation', linewidth=0.5)
        ax[1].plot(wavelengths, spectr2[i], label='GT Spectrum', linewidth=0.5)
        simulated_attenuation  = M @ (M_pinv @ attenuation[i])
        ax[0].plot(wavelengths, simulated_attenuation, label='Simulated Attenuation (No Scatter)', linewidth=3, alpha=0.9)
        simulated_spectrum = np.exp(-simulated_attenuation)*spectr1
        ax[1].plot(wavelengths, simulated_spectrum, label='Simulated Spectrum (No Scatter)', linewidth=3, alpha=0.9)
        scatter_simulated_attenuation = f(np.append(coef_list[i], scattering_params_list[i]), 0, x_min[1], x_min[0], M)
        ax[0].plot(wavelengths, scatter_simulated_attenuation, label='Simulated Attenuation (Scatter b_1(t))', linewidth=3, alpha=0.9)
        scatter_simulated_spectrum = np.exp(-scatter_simulated_attenuation)*spectr1
        ax[1].plot(wavelengths, scatter_simulated_spectrum, label='Simulated Spectrum (Scatter b_1(t))', linewidth=3, alpha=0.9)
        
        attenuation_error_noscatter += np.sum(np.abs(simulated_attenuation - attenuation[i])) / len(simulated_attenuation)
        attenuation_error_noscatter_780 += np.sum(np.abs(simulated_attenuation[x < 780] - attenuation[i][x < 780])) / len(simulated_attenuation[x < 780])

        attenuation_error_scatter += np.sum(np.abs(scatter_simulated_attenuation - attenuation[i])) / len(scatter_simulated_attenuation)
        attenuation_error_scatter_780 += np.sum(np.abs(scatter_simulated_attenuation[x < 780] - attenuation[i][x < 780])) / len(scatter_simulated_attenuation[x < 780])
        
        # spectrum_error_noscatter += np.sum(np.abs(simulated_spectrum - spectr2[i])) / len(simulated_spectrum)
        # spectrum_error_noscatter_780 += np.sum(np.abs(simulated_spectrum[x < 780] - spectr2[i][x < 780])) / len(simulated_spectrum[x < 780])
        
        # spectrum_error_scatter += np.sum(np.abs(scatter_simulated_spectrum - spectr2[i])) / len(scatter_simulated_spectrum)
        # spectrum_error_scatter_780 += np.sum(np.abs(scatter_simulated_spectrum[x < 780] - spectr2[i][x < 780])) / len(scatter_simulated_spectrum[x < 780])

        assert(len(simulated_attenuation) == len(scatter_simulated_attenuation) == len(simulated_spectrum) == len(scatter_simulated_spectrum))
        assert(len(simulated_attenuation[x < 780]) == len(scatter_simulated_attenuation[x < 780]) == len(simulated_spectrum[x < 780]) == len(scatter_simulated_spectrum[x < 780]))
        
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title("Attenuations at timepoint t_i = " + str(i))
        ax[1].set_title("Spectra at timepoint t_i = " + str(i))
        
        ax[0].set_xlabel("Wavelength")
        ax[1].set_xlabel("Wavelength")
        ax[0].set_ylabel("Attenuation")
        ax[1].set_ylabel("Intensity")
        
        if i % 50 == 0:
            fig.savefig(directory+'/'+str(i))
        plt.close()
            
        n+=1
        
    assert(n==1000)
    attenuation_error_noscatter /= n
    attenuation_error_scatter /= n    
    # spectrum_error_noscatter /= n
    # spectrum_error_scatter /= n
    
    attenuation_error_noscatter_780 /= n
    attenuation_error_scatter_780 /= n    
    # spectrum_error_noscatter_780 /= n
    # spectrum_error_scatter_780 /= n
    
    error_results = [attenuation_error_noscatter, attenuation_error_scatter, attenuation_error_noscatter_780, attenuation_error_scatter_780]
    #error_results = [spectrum_error_noscatter, spectrum_error_scatter, spectrum_error_noscatter_780, spectrum_error_scatter_780]
    with open(directory + '/attenuation_spectra_errors.pkl', 'wb') as pickle_file:
        pickle.dump(error_results, pickle_file)
    print(error_results)
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    # labels = ['Attenuation', 'Spectra']
    labels = ['Complete', 'Below 780nm']
    
    # linear_model = [attenuation_error_noscatter, spectrum_error_noscatter]
    # nonlinear_scatter_model = [attenuation_error_scatter, spectrum_error_scatter]
    
    # linear_model_780 = [attenuation_error_noscatter_780, spectrum_error_noscatter_780]
    # nonlinear_scatter_model_780 = [attenuation_error_scatter_780, spectrum_error_scatter_780]
    
    linear_model = [attenuation_error_noscatter, attenuation_error_noscatter_780]
    nonlinear_scatter_model = [attenuation_error_scatter, attenuation_error_scatter_780]
    
    #y_max = max(linear_model + nonlinear_scatter_model + linear_model_780 + nonlinear_scatter_model_780)
    y_max = max(linear_model + nonlinear_scatter_model)
    
    x_axis = np.arange(len(labels))
    
    width = 0.35
    rects1 = ax.bar(x_axis - width/2, linear_model, width, label='Standard Model')
    rects2 = ax.bar(x_axis + width/2, nonlinear_scatter_model, width, label='Scattering Model')
    
    # rects1 = ax[1].bar(x_axis - width/2, linear_model_780, width, label='Standard Model')
    # rects2 = ax[1].bar(x_axis + width/2, nonlinear_scatter_model_780, width, label='Scattering Model')
    
    ax.set_title("Mean Absolute Attenuation Error for Piglet "+pig)
    # ax[1].set_title("Mean Absolute Error below 780nm for Piglet "+pig+'_'+str(date))

    # for ax in ax:
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Spectral Range")
    ax.set_ylabel("Mean Abs. Error")
    ax.legend()
    ax.set_ylim(0, y_max*1.1)
        
    fig.savefig(directory+'/'+'attenuations_spectra_error')
        
    fig, ax = plt.subplots(1,4,figsize=(32,6))
    coef_list_v = np.vstack(coef_list)
    paper_concentrations_v = np.vstack(paper_concentrations)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    coef = ["HbO2", "Hbb","diffCCO"]
    
    coef_paper = ["Hbb","HbO2","diffCCO"]
    colors_paper = ['#ff7f0e', '#1f77b4', '#2ca02c']
    
    concentrations_no_scatter = M_pinv @ np.swapaxes(b, 0, 1)
    #print(concentrations_no_scatter.shape)
    for i in range(config.molecule_count):
        ax[0].plot(paper_concentrations_v[:1000,i],label=coef_paper[i], color=colors_paper[i], linewidth=2)
    #ax[0].plot(coef_list_v[:, 0] + coef_list_v[:,1], '--', color='#d62728', label="HbT",linewidth=2)
    
    plot_concentrations(ax[1], concentrations_no_scatter, coef_list_v)
    
    scattering_params_np = np.vstack(scattering_params_list)
    ax[2].plot(scattering_params_np[:,0])
    ax[2].set_xlabel("Timepoint t_i")
    ax[2].set_ylabel("a(t_i)")
    ax[2].set_title("Inferred a(t) in Scattering Model")
    ax[3].plot(scattering_params_np[:,1])
    ax[3].set_xlabel("Timepoint t_i")
    ax[3].set_ylabel("b(t_i)")
    ax[3].set_title("Inferred b(t) in Scattering Model")
    
    
    ax[0].legend()
    ax[0].set_xlabel("Timepoint t_i")
    ax[0].set_ylabel("Concentration")
    ax[0].set_title("Concentrations Inferred in Dataset")
    fig.savefig(directory+'/'+"concentrations")
    
    plt.clf()
    plt.close('all')
    #print(np.array(coef_list).shape)
    b = np.swapaxes(b, 0, 1)
    return np.transpose(np.array(coef_list)), b, scattering_params_list, x_min, time_taken

    #################################

    # b = spectr2 / spectr1
    # b = np.log(1 / np.asarray(b))  # see the writting above (we took log of I_R and there was also minus that went to the degree of the logarithm)
    # X = cp.Variable((m, len(b)))
    # b = np.swapaxes(b, 0, 1)
    
    # print(M.shape, X.shape, b.shape)
     
    # objective = cp.Minimize(cp.sum_squares(M @ X - b))
    # constraints = []
    # prob = cp.Problem(objective, constraints)
    
    # start = time.time()
    # result = prob.solve()
    # print("Time:", time.time() - start)
    # print(X.value.shape)
    # return -X.value, b
    
def optimisation_ti_error_timing(params_t1, *args):
    error = optimisation_ti_timing(params_t1, *args)
    return error
    
def jac(X,*arg):
    b = arg[0]
    #b_t1, a_t1 = arg[1], arg[2]
    M = arg[3]
    m = config.molecule_count
    delta_c_i = X[:m] 
    a_1 = X[m:(m+1)]
    b_1 = X[(m+1):(m+2)]  
    
    J = np.zeros((len(b), m+2))
    J[:,0:m]=M[:,0:m]
    for i in range(J.shape[0]):
        # J[i,0] = M[i,0]
        # J[i,1] = M[i,1]
        # J[i,2] = M[i,2]
        J[i,m] = ((wavelengths[i]/500)**(-b_1)) / (1-0.9)
        J[i,m+1] = ((((wavelengths[i]/500)**(-b_1))*a_1) / (1-0.9)) * (-np.log(wavelengths[i]/500))
    return J    
    
def optimisation_ti_timing(params_t1,*args):
    #print(params_t1)
    #print(len(args))
    
    a_t1, b_t1 = params_t1[0], params_t1[1]
    b = args[0]
    M = args[1]
    wavelengths = args[2]
    left_bound = np.append(np.ones(config.molecule_count)*(-np.inf), [-np.inf, 0])
    right_bound = np.append(np.ones(config.molecule_count)*np.inf, [np.inf, config.max_b])
    
    current_x = np.zeros(config.molecule_count+2)
    current_x[-2] = a_t1
    current_x[-1] = b_t1
    
    #coef_list = []
    errors_scatter = 0
    
    for i in (range(0, b.shape[0])):
        result = scipy.optimize.least_squares(f, current_x, args=(b[i,:], b_t1, a_t1, M), bounds=(left_bound, right_bound))
        current_x = result.x
        #coef_list.append(result.x[:config.molecule_count])
        errors_scatter += np.sqrt(2*result.cost)
        
    #coef_list = np.vstack(coef_list)
    #plt.figure()
    #plt.plot(coef_list[:,0],label='Hbb')
    #plt.plot(coef_list[:,1],label='HbO2')
    #plt.plot(coef_list[:,2],label='diffCCO')
        
    error = errors_scatter / b.shape[0]
    return error


def optimisation_scattering_timing(b, M, wavelengths):
    # m = config.molecule_count  # number of parameters (from 2 to 6)
    # molecules, x = preprocessing.read_molecules(config.left_cut, config.right_cut, wavelengths)
    # y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _, y_cytoa_diff, _, _, _, _ = molecules
    # M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
    #                             np.asarray(y_hb_f),
    #                             np.asarray(y_coxa - y_creda))))
    
    # b = attenuations
    result = scipy.optimize.brute(optimisation_ti_error_timing, (slice(0,config.max_a, config.max_a/3), slice(0,config.max_b, config.max_b/3)), args=(b,M,wavelengths), finish=None, full_output=True, workers=1)


def optimisation_no_scattering(spectr1, spectr2, wavelengths):
    m = config.molecule_count  # number of parameters (from 2 to 6)
    molecules, x = preprocessing.read_molecules(config.left_cut, config.right_cut, wavelengths)
    y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _, y_cytoa_diff, _, _, _, _ = molecules
    
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa - y_creda))))
    M_pinv = pinv(M)
    #print(M_pinv.shape)
    b = spectr2 / spectr1
    b = np.log(1 / np.asarray(b))
    #print(b.shape)

    #result = scipy.optimize.shgo(optimisation_ti_error, [(0,max_a), (0,max_b)], args=(b,M), n=2, options={'maxfev':2}, sampling_method='sobol')
    #print(result)
    #error, coef_list = optimisation_ti(result.x,*(b,M))

    #coef_list = M_pinv @ np.transpose(b)
    #print(np.array(coef_list).shape)
    #b = np.swapaxes(b, 0, 1)
    b = np.swapaxes(b, 0, 1)
    return M_pinv @ b, b

    #################################

    # b = spectr2 / spectr1
    # b = np.log(1 / np.asarray(b))  # see the writting above (we took log of I_R and there was also minus that went to the degree of the logarithm)
    # X = cp.Variable((m, len(b)))
    # b = np.swapaxes(b, 0, 1)

    # print(M.shape, X.shape, b.shape)
        
    # objective = cp.Minimize(cp.sum_squares(M @ X - b))
    # constraints = []
    # prob = cp.Problem(objective, constraints)

    # start = time.time()
    # result = prob.solve()
    # print("Time:", time.time() - start)
    # print(X.value.shape)
    # return -X.value, b


def get_dataset(spectr, dark_full, white_full, wavelengths, pig, date,paper_concentrations):
    ref_spectr = (spectr[:, 0] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])

    spectra_list = []
    coef_list = []
    scattering_params_list = []

    comp_spectr = np.array([(spectr[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0]) for i in range(0,1000)])

    #comp_spectr = np.array(comp_spectr.tolist() * 11)
    if config.scattering_model:
        print("Optimizing with scattering")
        coef_diff, spect_diff, scattering_params_diff, t1, time_taken = optimisation_scattering(ref_spectr, comp_spectr, wavelengths,pig,date,paper_concentrations)
        scattering_params_list.append(scattering_params_diff)
    else:
        print("Optimizing without scattering")
        coef_diff, spect_diff = optimisation_no_scattering(ref_spectr, comp_spectr, wavelengths)

    spectra_list.append(spect_diff)
    coef_list.append(coef_diff)
    if config.scattering_model:
        utils.save_optimization_data(ref_spectr, spectra_list, coef_list, str(pig)+'_'+str(date), scattering_coef_list=scattering_params_list, t1=t1, time_taken=time_taken)
    else:
        utils.save_optimization_data(ref_spectr, spectra_list, coef_list, str(pig)+'_'+str(date))
    
if __name__ == "__main__":
    for pig in tqdm(dic.keys()):
        spectr, paper_concentrations = preprocess_piglet(pig)
        get_dataset(spectr, dark_full, white_full, wavelengths, pig,0,paper_concentrations)