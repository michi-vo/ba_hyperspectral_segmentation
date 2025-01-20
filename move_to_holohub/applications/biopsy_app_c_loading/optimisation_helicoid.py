#Choose patient
#patient = "012-02"
patient = "020-01"
patients = ["017-01", "016-04", "016-05", "012-01", "020-01", "015-01", "012-02", "025-02", "021-01", "008-01", "008-02", "010-03", "021-05", "014-01"]
opti_patients = ["012-01","020-01","008-02", "012-02", "015-01", "008-01", "016-04", "016-05", "025-02"]


#Specificy path to Helicoid, UCL-NIR-Spectra, CREATIS Spectra
path = "/m2/data/ba_hyperspectral_segmentation/src/dataset/hsi_brain_database/HSI_Human_Brain_Database_IEEE_Access"
dataset_path = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/"
path_absorp = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/UCL-NIR-Spectra/spectra/"
path_creatis = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/CREATIS-Spectra/spectra/"

from spectral import * 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bisect
import imageio as io
from PIL import Image
import torch
import numpy as np
import scipy
import random
import os
from tqdm import tqdm
from scipy.linalg import pinv
from utils import save_helicoid_optimization_data
import time
from config import left_cut_helicoid, right_cut_helicoid
from concurrent.futures import ProcessPoolExecutor
import config

left_cut = left_cut_helicoid
right_cut = right_cut_helicoid

def sfilter(x):
#    return medfilt(x)
    #return savgol_filter(x,5,1)
    return x

#os.chdir("..")
plot_dir = "./paper_plots/"

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

mpl.rc('image', cmap='terrain')
mpl.rcParams['text.color'] = 'grey'
mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['ytick.color'] = 'grey'
mpl.rcParams['axes.labelcolor'] = 'grey'
#plt.rcParams["font.family"] = "Laksaman"

def remove_ticks(ax):
    ax.set_xticks([]) 
    ax.set_yticks([]) 

hbo2_absorp = path_absorp + "hb02.txt"
hhb_absorp = path_absorp + "hb.txt"
water_absorp = path_absorp + "matcher94_nir_water_37.txt"
diff_cyto_absorp = path_absorp + "cytoxidase_diff_odmMcm.txt"
fat_absorp = path_absorp + "fat.txt"
cyto_oxy_absorp = path_absorp + "moody cyt aa3 oxidised.txt"
cyto_red_absorp = path_absorp + "moody cyt aa3 reduced.txt"
### Using the preprocessing function we know
def load_spectra():
    path_dict = {}
    path_dict["cytoa_oxy"] = path_absorp + "moody cyt aa3 oxidised.txt"
    path_dict["cytoa_red"] = path_absorp + "moody cyt aa3 reduced.txt"
    path_dict["hbo2"] = path_absorp + "hb02.txt"
    path_dict["hbo2_450"] = path_absorp + "z_adult_hbo2_450_630.txt"
    path_dict["hbo2_600"] = path_absorp + "z_adult_hbo2_600_800.txt"
    path_dict["hb"] = path_absorp + "hb.txt"
    path_dict["hb_450"] = path_absorp + "z_adult_hb_450_630.txt"
    path_dict["hb_600"] = path_absorp + "z_adult_hb_600_800.txt"
    path_dict["water"] = path_absorp + "matcher94_nir_water_37.txt"
    path_dict["fat"] = path_absorp + "fat.txt"
    path_dict["water_hale"] = path_absorp + "water_hale73.txt"

    path_dict["cytoa_diff"] = path_absorp + "cytoxidase_diff_odmMcm.txt"
    path_dict["cytoc_oxy"] = path_absorp + "cooper pig c oxidised.txt"
    path_dict["cytoc_red"] = path_absorp + "cooper pig c reduced.txt"
    path_dict["cytob_oxy"] = path_absorp + "cope cyt b oxidised.txt"
    path_dict["cytob_red"] = path_absorp + "cope cyt b reduced.txt"

    return path_dict


### reading cpectra from .txt
def read_spectra(file_name):
    with open(file_name, 'r') as data:
        x, y = [], []
        for line in data:
            p = line.split()
            if not p[0] == '\x00':
                x.append(float(p[0]))
                y.append(float(p[1]))
    return np.array(x), np.array(y)

def read_spectra_creatis(filename):
    with open(filename, 'r') as file:
        return np.array(file.read().split(), dtype=float)

def cut_spectra(x, y, left_cut, right_cut):
    """
    cuts off spectrogram according to cut values
    """
    #print(x)
    ix_left = np.where(x >= left_cut)[0][0]
    ix_right = np.where(x >= right_cut)[0][0]
    #print("ix_left", ix_left)
    #print("ix_right", ix_right)
    return y[ix_left:ix_right + 1]


def wave_interpolation(y, x, mol_list, x_waves):
    """
    interpolate spectrogram values according to x_waves
    """
    lower_bound, upper_bound = x[0], x[-1]
    new_x = np.asarray([i for i in x_waves if lower_bound < i < upper_bound])

    new_y = {}
    for i in mol_list:
        #print(i, new_x.shape, x.shape, y[i].shape)
        new_y[i] = np.interp(new_x, x, y[i])

    return new_y, new_x

def read_molecules(left_cut, right_cut, mol_list=None, x_waves=None):
    #print(x_waves)
    path_dict = load_spectra()

    # read spectra for: cytochrome oxydised/reduced, oxyhemoglobin, hemoglobin, water, fat
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600", "water", "fat" ,"cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600", "water", "fat"]
    
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600"]
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600","cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600","cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red", "fat"]
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600","cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red", "fat", "water"]
    if mol_list == None:
        mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600","cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red","fat","water_hale"]

    
    x, y = {}, {}
    for i in mol_list:
        x[i], y[i] = read_spectra(path_dict[i])

    # from extinction to absorption
    # TODO check if water spectra was in extinction 
    
    if mol_list == None:
        y_list = ['hb_450', 'hb_600', 'hbo2_450', 'hbo2_600', 'cytoa_oxy', 'cytoa_red', "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
        
        for i in y_list:
            y[i] *= 2.3025851
    
        # from mm and micromole to cm and minimole, get rid of mole
        y["hbo2"] *= 10 * 1000 #/ 10
        y["hb"] *= 10 * 1000 #/ 10
    
        # from m to cm
        #y["fat"] /= 100
    
        for i in ["hbo2", "hb"]:
            xvals = np.array([i for i in range(int(x[i + "_450"][0]), int(x[i + "_600"][-1]) + 1)])
            yinterp = np.interp(xvals, np.concatenate([x[i + "_450"], x[i + "_600"]]), np.concatenate([y[i + "_450"], y[i + "_600"]]))
            x[i] = np.concatenate([xvals, x[i][151:]])
            y[i] = np.concatenate((yinterp, np.asarray(y[i][151:])), axis=None)
    
        x_new = x["cytoa_oxy"][bisect.bisect_left(x["cytoa_oxy"], left_cut):bisect.bisect_right(x["cytoa_oxy"], right_cut)]

    
    x_new = x[mol_list[0]][(x[mol_list[0]] >= left_cut) & (x[mol_list[0]] <= right_cut)]
    # print(x_new)
    # print(len(x_new))
    #print(x_new)
    for i in mol_list:
        y[i] = cut_spectra(x[i], y[i], left_cut, right_cut)
        #print(len(y[i]))

    if x_waves is not None:
        y, x_new = wave_interpolation(y, x_new, mol_list, x_waves)

    # print(len(x_new))
    # print(x_new)
    # print(len(y["hbo2"]))
    # print(len(y["hb"]))
    # print(len(y["cytoa_oxy"]))
    return [y[i] for i in mol_list], x_new

def read_molecules_cytochrome_cb(left_cut, right_cut, x_waves=None):
    #print(x_waves)
    path_dict = load_spectra()

    # read spectra for: cytochrome oxydised/reduced, oxyhemoglobin, hemoglobin, water, fat
    mol_list = ["cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]

    
    x, y = {}, {}
    for i in mol_list:
        x[i], y[i] = read_spectra(path_dict[i])

    # from extinction to absorption
    # TODO check if water spectra was in extinction 
    
    y_list = ["cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
    
    for i in y_list:
        y[i] *= 2.3025851
    
    # cutting all spectra to the range [left_cut, right_cut] nm
    x_new = x["cytoc_oxy"][(x["cytoc_oxy"] >= left_cut) & (x["cytoc_oxy"] <= right_cut)]
    mol_list = ["cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]

    for i in mol_list:
        y[i] = cut_spectra(x[i], y[i], left_cut, right_cut)
        #print(len(y[i]))

    if x_waves is not None:
        y, x_new = wave_interpolation(y, x_new, mol_list, x_waves)

    return [y[i] for i in mol_list], x_new

def read_molecules_creatis(left_cut, right_cut, x_waves=None):
    mol_list = ["eps_Hb", "eps_HbO2", "eps_oxCCO", "eps_redCCO", "mua_Fat", "mua_H2O"]
    wavelengths = read_spectra_creatis(path_creatis + "lambda.txt")
    y = {}
    for i in mol_list:
        y[i] = read_spectra_creatis(path_creatis+i+".txt")

    #print(y)
    # from extinction to absorption
    y_list = ["eps_Hb", "eps_HbO2", "eps_oxCCO", "eps_redCCO"]
    
    for i in y_list:
        y[i] *= 2.3025851
        y[i] /= 1000 #to cm^-1 / mM (millimole)
    
    # cutting all spectra to the range [left_cut, right_cut] nm
    x_new = wavelengths[(wavelengths >= left_cut) & (wavelengths <= right_cut)]

    # print(x_new)
    # print(len(x_new))
    #print(x_new)
    for i in mol_list:
        y[i] = y[i][(wavelengths >= left_cut) & (wavelengths <= right_cut)]
        y[i][y[i]<0]=0
        #print(len(y[i]))

    if x_waves is not None:
        y, x_new = wave_interpolation(y, x_new, mol_list, x_waves)

    # print(len(x_new))
    # print(x_new)
    # print(len(y["hbo2"]))
    # print(len(y["hb"]))
    # print(len(y["cytoa_oxy"]))
    return [y[i] for i in mol_list], x_new

def S(a_ti, b_ti, b_t1, a_t1, x):
        #print(x)
        result = ((((x/500)**(-b_ti)) * a_ti) - (((x/500)**(-b_t1)) * a_t1)) / (1-0.9)
        #result = ((((x/500)**(-b_t1)) * a_ti) - (((x/500)**(-b_t1)) * a_t1)) / (1-0.9)
        #result = ((((np.power(np.divide(x,500),-b_t1))) * a_ti) - (((np.power(np.divide(x,500),-b_t1))) * a_t1)) / (1-0.9)
        return result

def f(X,*arg):
    b = arg[0]
    b_t1, a_t1 = arg[1], arg[2]
    M = arg[3]
    x = arg[4]
    m = M.shape[1]
    delta_c_i = X[:m]
    a_ti = X[m:(m+1)]
    b_ti = X[(m+1):(m+2)]  
    
    #print("M shape", M.shape)
    #print("del c shape", delta_c_i.shape)
    
    result = (M @ delta_c_i) + S(a_ti, b_ti, b_t1, a_t1,x) - b
    #scattering = ((((np.power(np.divide(x,500),-b_t1))))) / (1-0.9)
    #print(M.shape)
    #print(scattering.shape)
    #M_ext = np.hstack((M, scattering[:, np.newaxis]))
    #result = (M_ext @ X[:(m+1)]) - b
    return result

def helicoid_optimisation_ti(params_t1,*args):
    #print(params_t1)
    #print(len(args))
    a_t1, b_t1 = params_t1[0], params_t1[1]
    b = args[0]
    M = args[1]
    x = args[2]
    left_bound = np.append(np.ones(M.shape[1])*(-np.inf), [-np.inf, 0])
    right_bound = np.append(np.ones(M.shape[1])*np.inf, [np.inf, config.max_b])
    
    current_x = np.zeros(M.shape[1]+2)
    #current_x[-2] = 0
    current_x[-2] = a_t1
    current_x[-1] = b_t1
    
    #coef_list = []
    #scattering_params_list = []
    #errors_scatter = []
    coef_list = np.zeros((b.shape[0], M.shape[1]))
    scattering_params_list = np.zeros((b.shape[0], 2))
    errors_scatter = np.zeros((b.shape[0]))
    
    for i in (range(0, b.shape[0])):
        result = scipy.optimize.least_squares(f, current_x, args=(b[i], b_t1, a_t1, M, x), bounds=(left_bound, right_bound))
        #current_x = result.x
        coef_list[i,:] = (result.x[:M.shape[1]])
        scattering_params_list[i,:] = result.x[M.shape[1]:]
        errors_scatter[i] = np.sqrt(2*result.cost)
        
    print("One worker has finished its optimization")
    #error = sum(errors_scatter) / len(errors_scatter)
    error = np.mean(errors_scatter)
    return error, coef_list, scattering_params_list, errors_scatter
    
def helicoid_optimisation_ti_error(params_t1, *args):
    error, _, _, _ = helicoid_optimisation_ti(params_t1, *args)
    return error


def optimize_single_b(i, b_i, b_t1, a_t1, M, x, current_x, left_bound, right_bound):
    result = scipy.optimize.least_squares(f, current_x, args=(b_i, b_t1, a_t1, M, x), bounds=(left_bound, right_bound))
    coef = result.x[:M.shape[1]]
    scattering_params = result.x[M.shape[1]:]
    error_scatter = np.sqrt(2*result.cost)
    return coef, scattering_params, error_scatter

def helicoid_optimisation_ti_parallel(params_t1, *args):
    a_t1, b_t1 = params_t1
    b, M, x = args[:3]
    left_bound = np.append(np.ones(M.shape[1])*(-np.inf), [-np.inf, 0])
    right_bound = np.append(np.ones(M.shape[1])*np.inf, [np.inf, config.max_b])
    current_x = np.zeros(M.shape[1] + 2)
    current_x[-2], current_x[-1] = a_t1, b_t1

    coef_list = np.zeros((b.shape[0], M.shape[1]))
    scattering_params_list = np.zeros((b.shape[0], 2))
    errors_scatter = np.zeros(b.shape[0])
    print(b.shape[0])
    with ProcessPoolExecutor(128) as executor:
        results = executor.map(optimize_single_b, range(b.shape[0]), b, [b_t1]*b.shape[0], [a_t1]*b.shape[0], [M]*b.shape[0], [x]*b.shape[0], [current_x]*b.shape[0], [left_bound]*b.shape[0], [right_bound]*b.shape[0])
        
        for i, (coef, scattering_params, error_scatter) in tqdm(enumerate(results)):
            coef_list[i, :] = coef
            scattering_params_list[i, :] = scattering_params
            errors_scatter[i] = error_scatter

    error = np.mean(errors_scatter)
    return error, coef_list, scattering_params_list, errors_scatter

def helicoid_optimisation_scattering(img_normalized, reference_point, M, x, coarseness=1, wavelength=None, reference_signal=None):
    if reference_signal is not None:
        img_ref = -np.log(img_normalized[::coarseness,::coarseness,:] / reference_signal)[:,:,np.in1d(wavelength,x)]
    else:
        img_ref = -np.log(img_normalized[::coarseness,::coarseness,:] / img_normalized[reference_point[0], reference_point[1], :])[:,:,np.in1d(wavelength,x)]
    
    b = img_ref.reshape(-1, img_ref.shape[-1]) #reshape to 1D x wavelength
    #b = relative_attenuation[np.newaxis, :]
    #print(b.shape)
    
    directory = dataset_path + "helicoid/" + patient
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    M_pinv = pinv(M)
    # print(spectr2.shape)
    #b = spectr2 / spectr1
    #b = np.log(1 / np.asarray(b))
    start_time = time.time()
    result = scipy.optimize.brute(helicoid_optimisation_ti_error, (slice(0,config.max_a, config.max_a/3), slice(0,config.max_b, config.max_b/25)), args=(b,M,x), finish=None, full_output=True, workers=90)    
    #result = scipy.optimize.brute(helicoid_optimisation_ti_error, (slice(0,config.max_a, config.max_a/2), slice(0,config.max_b, config.max_b/2)), args=(b,M,x), finish=None, full_output=True, workers=60)    
    end_time = time.time()
    time_taken = end_time - start_time
    
    #result = scipy.optimize.brute(helicoid_optimisation_ti_error, (slice(0,config.max_a, config.max_a/3), slice(0,config.max_b, config.max_b/15)), args=(b,M,x), finish=None, full_output=True, workers=4)    
    #result = [[1,0.66],-1,-1,-1]
    
    #print(result)
    x_min = result[0]
    error_min = result[1]
    grid = result[2]
    error_grid = result[3]

    a_t1, b_t1 = x_min
    
    grid_x, grid_y = grid
    plt.figure()
    plt.pcolormesh(grid_x, grid_y, error_grid)
    clb = plt.colorbar()
    clb.set_label("Error")
    plt.xlabel("a(t_1)")
    plt.ylabel("b(t_1)")
    plt.savefig(directory+'/'+'error')
    
    error, coef_list, scattering_params_list, errors = helicoid_optimisation_ti(x_min,*(b,M,x))

    coef_list = np.array(coef_list)
    coef_list = coef_list.reshape(img_ref.shape[0], img_ref.shape[1], coef_list.shape[-1])
    scattering_params = np.array(scattering_params_list)
    scattering_params = scattering_params.reshape(img_ref.shape[0], img_ref.shape[1], scattering_params.shape[-1])

    errors = np.array(errors)
    errors = errors.reshape(img_ref.shape[0], img_ref.shape[1])
    
    print(coef_list.shape)
    print(scattering_params.shape)
    print(errors.shape)
    
    # fig, ax = plt.subplots(1,4,figsize=(32,6))
    # coef_list_v = np.vstack(coef_list)
    # paper_concentrations_v = np.vstack(paper_concentrations)
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    # coef = ["HbO2", "Hbb","diffCCO"]
    
    # coef_paper = ["Hbb","HbO2","diffCCO"]
    # colors_paper = ['#ff7f0e', '#1f77b4', '#2ca02c']
    
    # concentrations_no_scatter = M_pinv @ np.swapaxes(b, 0, 1)
    # #print(concentrations_no_scatter.shape)
    # for i in range(config.molecule_count):
    #     ax[0].plot(paper_concentrations_v[:1000,i],label=coef_paper[i], color=colors_paper[i], linewidth=2)
    #     ax[1].plot(concentrations_no_scatter[i, :], '--',label=coef[i] + ' (Standard Model)', color=colors[i], linewidth=2, alpha=0.8)
    #     ax[1].plot(coef_list_v[:, i], label=coef[i] + ' (Scattering Model)', color=colors[i], linewidth=2)
    # #ax[0].plot(coef_list_v[:, 0] + coef_list_v[:,1], '--', color='#d62728', label="HbT",linewidth=2)
    
    # scattering_params_np = np.vstack(scattering_params_list)
    # ax[2].plot(scattering_params_np[:,0])
    # ax[2].set_xlabel("Timepoint t_i")
    # ax[2].set_ylabel("a(t_i)")
    # ax[2].set_title("Inferred a(t) in Scattering Model")
    # ax[3].plot(scattering_params_np[:,1])
    # ax[3].set_xlabel("Timepoint t_i")
    # ax[3].set_ylabel("b(t_i)")
    # ax[3].set_title("Inferred b(t) in Scattering Model")
    
    
    # ax[0].legend()
    # ax[1].legend()
    # ax[0].set_xlabel("Timepoint t_i")
    # ax[1].set_xlabel("Timepoint t_i")
    # ax[0].set_ylabel("Concentration")
    # ax[1].set_ylabel("Concentration")
    # ax[0].set_title("Concentrations Inferred in Dataset")
    # ax[1].set_title("Inferred Concentrations")
    # fig.savefig(directory+'/'+"concentrations")
    
    # plt.clf()
    # plt.close('all')
    # #print(np.array(coef_list).shape)
    # b = np.swapaxes(b, 0, 1)
    #return np.transpose(np.array(coef_list)), b
    return coef_list, scattering_params, errors, a_t1, b_t1, img_ref, time_taken

def helicoid_optimisation_scattering_wlselection(img_normalized, reference_point, M, x, coarseness, patient, x_chosen, label_map=None, only_labeled=False, use_parallel=False):
    img_ref = -np.log(img_normalized[::coarseness,::coarseness,:] / img_normalized[reference_point[0], reference_point[1], :])
    
    
    img_ref_GT = torch.load("dataset/helicoid/" + patient + "/img_ref.pt").numpy()
    assert(np.allclose(img_ref_GT[::coarseness, ::coarseness, x_chosen], img_ref))
    coef_list_GT = torch.load("dataset/helicoid/" + patient + "/coef_list.pt").numpy()
    #print(coef_list_GT.shape)
    scattering_params_GT = torch.load("dataset/helicoid/" + patient + "/scattering_params.pt").numpy()
    
    coef_list_GT = coef_list_GT[::coarseness, ::coarseness, :]
    scattering_params_GT = scattering_params_GT[::coarseness, ::coarseness, :]

    b = img_ref.reshape(-1, img_ref.shape[-1])  # reshape to 1D x wavelength

    if only_labeled:
        label_map = label_map[::coarseness, ::coarseness, :]
        some_label_map = np.logical_or.reduce((label_map == 1, label_map == 2, label_map == 3))
        some_label_map = some_label_map.flatten()
        b = b[some_label_map,:]

    directory = dataset_path + "helicoid/" + patient
    if not os.path.exists(directory):
        os.makedirs(directory)
    M_pinv = np.linalg.pinv(M)
    # print(spectr2.shape)
    #b = spectr2 / spectr1
    #b = np.log(1 / np.asarray(b))
    x_min = torch.load("dataset/helicoid/" + patient + "/t1.pt").numpy()
    a_t1, b_t1 = x_min
    
    if use_parallel:
        error, coef_list, scattering_params_list, errors = helicoid_optimisation_ti_parallel(x_min, b, M, x)
    else:
        error, coef_list, scattering_params_list, _ = helicoid_optimisation_ti(x_min,*(b,M,x))

    coef_list = np.array(coef_list)
    coef_list = coef_list.reshape(img_ref.shape[0], img_ref.shape[1], coef_list.shape[-1])
    scattering_params = np.array(scattering_params_list)
    scattering_params = scattering_params.reshape(img_ref.shape[0], img_ref.shape[1], scattering_params.shape[-1])

    params_found = np.concatenate((coef_list, scattering_params), axis=2)
    params_found_GT = np.concatenate((coef_list_GT, scattering_params_GT), axis=2)

    #print(np.sqrt(np.mean((params_found - params_found_GT)**2, axis=2)))
    #print(np.mean(params_found_GT, axis=2))
    error_wavelength_selection = np.mean(np.sqrt(np.mean((params_found - params_found_GT)**2, axis=2) / np.mean(params_found_GT**2, axis=2)))
    #print(error_wavelength_selection)


    # errors = np.array(errors)
    # errors = errors.reshape(img_ref.shape[0], img_ref.shape[1])
    #return coef_list, scattering_params, errors, a_t1, b_t1, img_ref
    return error_wavelength_selection, params_found, params_found_GT

if __name__ == "__main__":
    for patient in opti_patients:
        hdr_path = path+"/{}/raw.hdr".format(patient)
        img = open_image(hdr_path)
        wavelength = np.array(img.metadata['wavelength']).astype(float)

        rbg_path = path+"/{}/image.jpg".format(patient)
        rbg = Image.open(rbg_path)

        gt_path = path+"/{}/gtMap.hdr".format(patient)
        gt = open_image(gt_path)
        gt = gt.load()
        print(gt.shape)

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
        print(img_normalized.shape)
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
        wavelength_index = {value: idx for idx, value in enumerate(wavelength)}
        wavelength_filtered = wavelength

        def find_wavelength(value):
            closest_value = min(wavelength, key=lambda x_val: abs(x_val - value))
            return wavelength_index[closest_value]

        img_RGB = np.stack((img_normalized[:,:,int(img.metadata['default bands'][2])],img_normalized[:,:,int(img.metadata['default bands'][1])],img_normalized[:,:,int(img.metadata['default bands'][0])]), axis=2)

        mol_list = ["water_hale"]
        molecules_ucl, x_ucl = read_molecules(left_cut, right_cut, mol_list, wavelength_filtered)

        molecules_cyto, x_cyto = read_molecules_cytochrome_cb(left_cut, right_cut, wavelength_filtered)

        molecules, x = read_molecules_creatis(left_cut, right_cut, x_waves=wavelength_filtered)

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

        config.molecule_count = M.shape[1]
        config.max_b = 4
        config.max_a = 100
        coarseness = 1

        hbt_opti = np.zeros((img.shape[0], img.shape[1]))
        hbdiff_opti = np.zeros((img.shape[0], img.shape[1]))
        error_opti = np.zeros((img.shape[0], img.shape[1]))
        diffCCO_opti = np.zeros((img.shape[0], img.shape[1]))

        reference_index_blood = random.randint(0, len(x_blood) - 1)
        reference_point = [x_blood[reference_index_blood], y_blood[reference_index_blood]]
        coefs, scattering_params, errors, a_t1, b_t1, img_ref, time_taken = helicoid_optimisation_scattering(img_normalized, reference_point, M, x)

        save_helicoid_optimization_data(img_ref, np.array(reference_point), coefs, scattering_params, errors, np.array([a_t1, b_t1]), time_taken, patient)

        img_ref_comp = -np.log((img_normalized[::coarseness,::coarseness,:] / img_normalized[reference_point[0], reference_point[1], :])[:,:,np.in1d(wavelength,x)])
        assert(np.all(img_ref == img_ref_comp))

        hbt_inferred = coefs[:,:,0] + coefs[:,:,1]
        hbdiff_inferred = coefs[:,:,0] - coefs[:,:,1]
        diffCCO_inferred = coefs[:,:,2] - coefs[:,:,3]
        blood_scattering_inferred = scattering_params[:,:,0]
        brain_scattering_inferred = scattering_params[:,:,1]
        error_inferred = errors

        img_RGB = np.stack((img_normalized[::coarseness,::coarseness,int(img.metadata['default bands'][2])],img_normalized[::coarseness,::coarseness,int(img.metadata['default bands'][1])],img_normalized[::coarseness,::coarseness,int(img.metadata['default bands'][0])]), axis=2)

        error_tolerance = 2.5
        (x_normal,y_normal) = np.where((np.squeeze(gt[::coarseness,::coarseness,:])==1) & (error_inferred < error_tolerance))
        (x_tumor,y_tumor) = np.where((np.squeeze(gt[::coarseness,::coarseness,:])==2) & (error_inferred < error_tolerance))
        (x_blood,y_blood) =  np.where((np.squeeze(gt[::coarseness,::coarseness,:])==3) & (error_inferred < error_tolerance))

        fig, axs = plt.subplots(nrows=4, ncols=3,figsize=(15,15))
        ax_RGB, ax_hbt, ax_cco, ax_bloodscatter, ax_brainscatter, ax_error, ax_greenpoint, ax_bluepoint, ax_yellowpoint, ax_bottom1, ax_bottom2, ax_bottom3 = axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2], axs[2,0], axs[2,1], axs[2,2], axs[3,0], axs[3,1], axs[3,2]
        ax_RGB.imshow(img_RGB)
        ax_RGB.set_title("Synthetic RGB", fontsize=20)
        #ax_RGB.imshow(mask1, cmap=cm.tab10)
        #ax_RGB.imshow(mask2, cmap=cm.Dark2)
        #ax_RGB.imshow(mask3, cmap=cm.get_cmap('spring_r'))

        hbt_filtered = hbt_inferred
        hbt_filtered = np.where(error_inferred > error_tolerance, np.min(hbt_inferred), hbt_filtered)
        #hbt_filtered = np.where(hbt_filtered < 0.0, 0, hbt_filtered)
        hbt_img = ax_hbt.imshow(hbt_filtered, cmap='Reds')
        fig.colorbar(hbt_img, ax=ax_hbt)
        #ax_hbt.set_title("Inferred HbT > 0 \n Fitting MAE < " + str(error_tolerance), fontsize=15)
        ax_hbt.set_title("Inferred HbT \n E < " + str(error_tolerance), fontsize=15)

        # CCO_inferred_filtered = np.where(error_inferred > error_tolerance, np.min(CCO_inferred), CCO_inferred)
        # CCO_inferred_filtered = np.where(CCO_inferred_filtered < 0, 0, CCO_inferred_filtered)
        # CCO_img = ax_cco.imshow(CCO_inferred_filtered)
        # fig.colorbar(CCO_img, ax=ax_cco)
        # ax_cco.set_title("CCO > 0(MAE < " + str(error_tolerance) +")")
        hbdiff_filtered = hbdiff_inferred
        hbdiff_filtered = np.where(error_inferred > error_tolerance, np.min(hbdiff_inferred), hbdiff_filtered)
        hbdiff_filtered = np.where(hbdiff_filtered < 0.0, 0, hbdiff_filtered)
        hbdiff_img = ax_cco.imshow(hbdiff_filtered, cmap='Reds')
        fig.colorbar(hbdiff_img, ax=ax_cco)
        #ax_hbt.set_title("Inferred HbT > 0 \n Fitting MAE < " + str(error_tolerance), fontsize=15)
        ax_cco.set_title("Inferred HbDiff > 0  \n E < " + str(error_tolerance), fontsize=15)

        #hbt_filtered_neg = hbdiff_inferred
        #hbt_filtered_neg = np.where(hbt_filtered_neg > 0.0, 0, hbt_filtered_neg)
        # hbt_filtered_neg = np.where(error_inferred > error_tolerance, 0, hbt_filtered_neg)
        # hbt_neg_img = ax_cco.imshow(hbt_filtered_neg)
        # fig.colorbar(hbt_neg_img, ax=ax_cco)
        # #ax_cco.set_title("Inferred HbDiff > 0\n Fitting MAE < " + str(error_tolerance), fontsize=15)
        # ax_cco.set_title("Inferred  \n Fitting MAE < " + str(error_tolerance), fontsize=15)

        blood_scattering_inferred_filtered = blood_scattering_inferred
        #blood_scattering_inferred_filtered = np.where(blood_scattering_inferred > 0.0, 0, blood_scattering_inferred)
        #blood_scattering_inferred_filtered = np.where(error_inferred > error_tolerance, np.nan, blood_scattering_inferred_filtered)
        blood_scattering_img = ax_bloodscatter.imshow((blood_scattering_inferred_filtered), cmap='seismic')
        ax_bloodscatter.set_title("Inferred a_ti", fontsize=13)
        fig.colorbar(blood_scattering_img, ax=ax_bloodscatter)

        brain_scattering_inferred_filtered = brain_scattering_inferred
        #brain_scattering_inferred_filtered = np.where(brain_scattering_inferred_filtered < 0.0, 0, brain_scattering_inferred_filtered)
        #brain_scattering_inferred_filtered = np.where(error_inferred > error_tolerance, np.nan, brain_scattering_inferred_filtered)
        brain_scattering_img = ax_brainscatter.imshow((brain_scattering_inferred_filtered), cmap='seismic_r')
        #ax_brainscatter.set_title("Inferred brain scattering", fontsize=13)
        ax_brainscatter.set_title("Inferred b_ti", fontsize=13)
        fig.colorbar(brain_scattering_img, ax=ax_brainscatter)

        # # error_img = ax_error.imshow(error_inferred)
        # # fig.colorbar(error_img, ax=ax_error)
        # # ax_error.set_title("MAE of Inference", fontsize=15)
        # tumor_scattering_inferred_filtered = tumor_scattering_inferred
        # #tumor_scattering_inferred_filtered = np.where(error_inferred > error_tolerance, np.nan, tumor_scattering_inferred_filtered)
        # error_img = ax_error.imshow(tumor_scattering_inferred_filtered, cmap='seismic_r')
        # fig.colorbar(error_img, ax=ax_error)
        # ax_error.set_title("Inferred tumor scattering", fontsize=15)

        diffCCO_inferred_filtered = np.where(error_inferred > error_tolerance, np.min(diffCCO_inferred), diffCCO_inferred)
        #diffCCO_inferred_filtered = np.where(diffCCO_inferred_filtered < 0, 0, diffCCO_inferred_filtered)
        #diffCCO_img = ax_error.imshow(diffCCO_inferred_filtered, norm=Normalize(vmin=0, vmax=diffCCO_inferred_filtered.max()/4))
        diffCCO_img = ax_error.imshow(diffCCO_inferred_filtered)
        fig.colorbar(diffCCO_img, ax=ax_error)
        ax_error.set_title("diffCCO (E < " + str(error_tolerance) +")")

        rows, cols = np.indices(error_inferred.shape)
        #points_1 = np.where((error_inferred <= error_tolerance*0.25) & (hbt_inferred > 0.005) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
        #points_1 = np.where((error_inferred <= error_tolerance) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
        green_index = random.randint(0, len(x_normal) - 1)
        green_point = [x_normal[green_index], y_normal[green_index]]

        yellow_index = random.randint(0, len(x_blood) - 1)
        yellow_point = [x_blood[yellow_index], y_blood[yellow_index]]

        ax_RGB.plot(green_point[1], green_point[0], marker="o", markersize=5, markeredgecolor="darkgreen", markerfacecolor='#66FF66')
        ax_RGB.plot(yellow_point[1], yellow_point[0], marker="o", markersize=5, markeredgecolor="yellow", markerfacecolor='yellow')
        #points_1[0].shape

        #points_2 = np.where((hbt_inferred > 0.005) & (error_inferred >= error_tolerance*0.9) & (error_inferred <= error_tolerance) & (hbt_inferred > 0.005) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
        #points_2 = np.where((error_inferred >= error_tolerance*0.75) & (error_inferred <= error_tolerance) & (rows > 150) & (rows < 250) & (cols > 150) & (cols < 250))
        #orange_point = [points_2[0][0], points_2[1][0]]

        #(x_normal,y_normal) = np.where((np.squeeze(gt)==1) & (error_inferred < error_tolerance))
        #(x_tumor,y_tumor) = np.where((np.squeeze(gt)==2) & (error_inferred < error_tolerance))

        (x_normal,y_normal) = np.where((np.squeeze(gt[::coarseness,::coarseness,:])==1) & (error_inferred < error_tolerance))
        (x_tumor,y_tumor) = np.where((np.squeeze(gt[::coarseness,::coarseness,:])==2) & (error_inferred < error_tolerance))
        (x_blood,y_blood) =  np.where((np.squeeze(gt[::coarseness,::coarseness,:])==3) & (error_inferred < error_tolerance))

        #ax_histo.hist(oxyCCO_inferred[x_normal,y_normal], bins=30, alpha=0.5, label='diffCCO (Normal)', density=True)
        #ax_histo.hist(oxyCCO_inferred[x_tumor,y_tumor], bins=30, alpha=0.5, label='diffCCO (Tumor)', density=True)

        #ax_histo.hist(tumor_scattering_inferred[x_normal,y_normal], bins=30, alpha=0.5, label='diffCCO (Normal)', density=True)
        #ax_histo.hist(tumor_scattering_inferred[x_tumor,y_tumor], bins=30, alpha=0.5, label='diffCCO (Tumor)', density=True)

        ax_bottom1.hist(hbt_inferred[x_normal,y_normal], bins=30, alpha=0.5, label='HbT (Normal)')
        print(len(x_normal))
        ax_bottom1.hist(hbt_inferred[x_tumor,y_tumor], bins=30, alpha=0.5, label='HbT (Tumor)')
        print(len(x_tumor))
        ax_bottom1.hist(hbt_inferred[x_blood,y_blood], bins=30, alpha=0.5, label='HbT (Blood)')
        ax_bottom1.legend()

        # ax_bottom2.hist(CCO_inferred[x_normal,y_normal], bins=30, alpha=0.5, label='CCO (Normal)')
        # ax_bottom2.hist(CCO_inferred[x_tumor,y_tumor], bins=30, alpha=0.5, label='CCO (Tumor)')
        # ax_bottom2.legend()

        ax_bottom2.hist(hbdiff_inferred[x_normal,y_normal], bins=30, alpha=0.5, label='HbDiff (Normal)')
        ax_bottom2.hist(hbdiff_inferred[x_tumor,y_tumor], bins=30, alpha=0.5, label='HbDiff (Tumor)')
        ax_bottom2.hist(hbdiff_inferred[x_blood,y_blood], bins=30, alpha=0.5, label='HbDiff (Blood)')
        ax_bottom2.legend()

        ax_bottom3.hist(diffCCO_inferred[x_normal,y_normal], bins=30, alpha=0.5, label='diffCCO (Normal)')
        ax_bottom3.hist(diffCCO_inferred[x_tumor,y_tumor], bins=30, alpha=0.5, label='diffCCO (Tumor)')
        ax_bottom3.hist(diffCCO_inferred[x_blood,y_blood], bins=30, alpha=0.5, label='diffCCO (Blood)')
        ax_bottom3.set_title("diffCCO")
        # ax_bottom3.hist(CCO_inferred[x_normal,y_normal], bins=30, alpha=0.5, label='CCO (Normal)')
        # ax_bottom3.hist(CCO_inferred[x_tumor,y_tumor], bins=30, alpha=0.5, label='CCO (Tumor)')
        # ax_bottom3.hist(CCO_inferred[x_blood,y_blood], bins=30, alpha=0.5, label='CCO (Blood)')
        # ax_bottom3.set_title("CCO")
        ax_bottom3.legend()

        #ax_RGB.plot(reference_point[1], reference_point[0], marker="x", markersize=8, markeredgecolor="orange", markerfacecolor="#FFA500")


        relative_attenuation = img_ref[green_point[0], green_point[1], :]
        print(relative_attenuation.shape)
        green_params = np.concatenate((coefs[green_point[0], green_point[1], :].flatten(), scattering_params[green_point[0],green_point[1],:].flatten()))
        print(M.shape)
        inferred_attenuation = f(green_params, 0, b_t1, a_t1, M, x)
        ax_greenpoint.plot(x, relative_attenuation, label='GT')
        ax_greenpoint.plot(x, inferred_attenuation, linewidth=4, label='BLL')
        ax_greenpoint.legend()
        #print(error_inferred[green_point[0], green_point[1]])
        #print(np.mean(np.abs((inferred_attenuation - relative_attenuation))))
        assert np.isclose(error_inferred[green_point[0], green_point[1]],np.sqrt(np.sum(np.square((inferred_attenuation - relative_attenuation)))))
        ax_greenpoint.set_title("Green point with MAE " + str(round(error_inferred[green_point[0], green_point[1]],3)) + " (Normal)", fontsize=12)

        if len(x_tumor) > 0:
            blue_index = random.randint(0, len(x_tumor) - 1)
            blue_point = [x_tumor[blue_index], y_tumor[blue_index]]
            ax_RGB.plot(blue_point[1], blue_point[0], marker="o", markersize=5, markeredgecolor="darkblue", markerfacecolor="#0000FF")
            relative_attenuation = img_ref[blue_point[0], blue_point[1], :]
            blue_params = np.concatenate((coefs[blue_point[0], blue_point[1], :].flatten(), scattering_params[blue_point[0],blue_point[1],:].flatten()))
            inferred_attenuation = f(blue_params, 0, b_t1, a_t1, M, x)
            ax_bluepoint.plot(x, relative_attenuation, label='GT')
            ax_bluepoint.plot(x, inferred_attenuation, linewidth=4, label='BLL')
            ax_bluepoint.legend()
            assert np.isclose(error_inferred[blue_point[0], blue_point[1]],np.sqrt(np.sum(np.square((inferred_attenuation - relative_attenuation)))))
            ax_bluepoint.set_title("Blue point with MAE " + str(round(error_inferred[blue_point[0], blue_point[1]],3)) + " (Tumor)", fontsize=12)

        relative_attenuation = img_ref[yellow_point[0], yellow_point[1], :]
        yellow_params = np.concatenate((coefs[yellow_point[0], yellow_point[1], :].flatten(), scattering_params[yellow_point[0], yellow_point[1],:].flatten()))
        inferred_attenuation = f(yellow_params, 0, b_t1, a_t1, M, x)
        ax_yellowpoint.plot(x, relative_attenuation, label='GT')
        ax_yellowpoint.plot(x, inferred_attenuation, linewidth=4, label='BLL')
        ax_yellowpoint.legend()
        assert np.isclose(error_inferred[yellow_point[0], yellow_point[1]], np.sqrt(np.sum(np.square((inferred_attenuation - relative_attenuation)))))
        ax_yellowpoint.set_title("Yellow point with MAE " + str(round(error_inferred[yellow_point[0], yellow_point[1]],3)) + " (Blood)", fontsize=12)

        fig.savefig(patient+"_scatter")