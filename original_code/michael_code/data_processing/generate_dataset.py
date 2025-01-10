import pickle
import os

import numpy as np
import scipy
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from spectral import *
import config
import data_processing.preprocessing as prepro
import utils
#from config import left_cut, right_cut, molecule_count, max_b, max_a
from utils import beerlamb_scattering_delta_attenuation, beerlamb_delta_attenuation
from optimisation_helicoid import read_molecules_creatis, read_molecules_cytochrome_cb
from optimisation_helicoid import path, left_cut, right_cut
from config import left_cut_helicoid, right_cut_helicoid


def load_Xwaves():
    """
    loads the available wavelengths from one example dataset (one piglet). !might not be consistent across all samples!
    """
    # path = "dataset/HSI_Human_Brain_Database_IEEE_Access/{}/raw.hdr".format('004-02')
    # x_waves = prepro.read_wavelengths(path)
    x_waves = scipy.io.loadmat(config.dataset_path + 'LWP483_10Jan2017_SharedHyperProbe.mat')['wavelengths'].astype(float)
    molecules, x = prepro.read_molecules(config.left_cut, config.right_cut, x_waves)
    return molecules, x


def save_data(syn_data, syn_params, is_scattering_model, name=""):
    """
    helper method to save synthetic dataset
    """
    #path_spectra = 'synthetic/spectra'
    #path_params = 'synthetic/params'
    if is_scattering_model:
        print("Saving scattering model")
        path = config.dataset_path + "synthetic_scattering/"
        path_spectra = path + 'spectra'
        path_params = path + 'params'
    else:
        print("Saving no scattering model")
        path = config.dataset_path + "synthetic_no_scattering/"
        path_spectra = path + 'spectra'
        path_params = path + 'params'  
        
    if not os.path.exists(path): os.mkdir(path)
    
    with open(path_spectra + name + '.pkl', 'wb') as f:
        pickle.dump(torch.from_numpy(np.array(syn_data)).float(), f)
    with open(path_params + name + '.pkl', 'wb') as f:
        pickle.dump(torch.from_numpy(np.array(syn_params)).float(), f)

def save_data_helicoid(syn_data, syn_params):
    """
    helper method to save synthetic dataset
    """
    #path_spectra = 'synthetic/spectra'
    #path_params = 'synthetic/params'
    path = config.dataset_path + "/helicoid/synthetic_scattering/"
    path_spectra = path + 'spectra'
    path_params = path + 'params'
        
    if not os.path.exists(path): os.mkdir(path)
    
    with open(path_spectra + '.pkl', 'wb') as f:
        pickle.dump(torch.from_numpy(np.array(syn_data)).float(), f)
    with open(path_params + '.pkl', 'wb') as f:
        pickle.dump(torch.from_numpy(np.array(syn_params)).float(), f)

def get_params():
    """
    sample a random set of molecule concentrations (according to some rules)
    """
    #params = np.random.dirichlet(np.ones(config.molecule_count), size=1)[0]

    Hbb = np.random.dirichlet(np.ones(2), size=1)[0]
    oxyCCO = np.random.uniform(low=0.0, high=0.2)
    redCCO = np.random.uniform(low=0.0, high=0.2)
    Hbb *= (1 - (redCCO + oxyCCO))
    params = np.array([Hbb[0], Hbb[1], oxyCCO, redCCO])

    return params

def get_scattering_params():
    b_1_max = config.max_b
    a_1_max= config.max_a
    params = 2*(np.array(np.random.rand(3)))-1 #gets random params for HbO2, Hbb, diffCCO in the range [-0.5,0.5]
    params[2] = 0.5*params[2]
    a_1 = np.random.rand()*a_1_max
    b_1 = np.random.rand()*b_1_max
    a_ti = a_1 + a_1_max*0.1*(np.random.rand() - 0.5)
    b_ti = b_1 + b_1_max*0.1*(np.random.rand() - 0.5)
    return np.concatenate([params, np.array([a_1, b_1, a_ti, b_ti])])

def get_no_scattering_params():
    params = np.array(np.random.rand(3))-0.5 #gets random params for HbO2, Hbb, diffCCO in the range [-0.5,0.5]
    params[2] = 0.5*params[2]
    return params

def get_scattering_params_helicoid(minsmaxs):
    params = np.zeros(10)
    params[0] = np.random.uniform(minsmaxs['HbO2']['min'], minsmaxs['HbO2']['max'])
    params[1] = np.random.uniform(minsmaxs['Hbb']['min'], minsmaxs['Hbb']['max'])
    params[2] = np.random.uniform(minsmaxs['oxCCO']['min'], minsmaxs['oxCCO']['max'])
    params[3] = np.random.uniform(minsmaxs['redCCO']['min'], minsmaxs['redCCO']['max'])
    
    params[4] = np.random.uniform(minsmaxs['oxCC']['min'], minsmaxs['oxCC']['max'])
    params[5] = np.random.uniform(minsmaxs['redCC']['min'], minsmaxs['redCC']['max'])
    params[6] = np.random.uniform(minsmaxs['oxCB']['min'], minsmaxs['oxCB']['max'])
    params[7] = np.random.uniform(minsmaxs['redCB']['min'], minsmaxs['redCB']['max'])
    
    params[8] = np.random.uniform(minsmaxs['water']['min'], minsmaxs['water']['max'])
    params[9] = np.random.uniform(minsmaxs['fat']['min'], minsmaxs['fat']['max'])
    
    a_1 = np.random.uniform(minsmaxs['a_1']['min'], minsmaxs['a_1']['max'])
    b_1 = np.random.uniform(minsmaxs['b_1']['min'], minsmaxs['b_1']['max'])
    
    a_ti = np.max(a_1 + np.random.uniform(minsmaxs['diff_a_t']['min'], minsmaxs['diff_a_t']['max']), 0)
    b_ti = np.max(b_1 + np.random.uniform(minsmaxs['diff_b_t']['min'], minsmaxs['diff_b_t']['max']), 0)
    
    # a_ti = np.random.uniform(minsmaxs['a_ti']['min'], minsmaxs['a_ti']['max'])
    # b_ti = np.random.uniform(minsmaxs['b_ti']['min'], minsmaxs['b_ti']['max'])
    
    return np.concatenate([params, np.array([a_1, b_1, a_ti, b_ti])])

# def generate_dataset(n_samples=10000):
#     """
#     generates synthetic dataset of (spectrogram, param) pairs with n_samples
#     """
#     molecules, x = load_Xwaves()

#     syn_data, syn_params = None, None
#     for i in tqdm(range(n_samples)):
#         params = get_params()
#         if i == 0:
#             syn_data = np.expand_dims(utils.beerlamb_multi(molecules, x, params, config.left_cut), axis=0)
#             syn_params = np.expand_dims(params, axis=0)
#         else:
#             syn_data = np.concatenate([syn_data, np.expand_dims(utils.beerlamb_multi(molecules, x, params, config.left_cut), axis=0)])
#             syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
#     save_data(syn_data, syn_params)


# def generate_diff_dataset(n_samples=100000):
#     """
#     generates synthetic dataset of ((spectrogram - ref_spectrogram), params2-params) pairs with n_samples
#     """
#     molecules, x = load_Xwaves()

#     syn_data, syn_params = None, None
#     for i in tqdm(range(n_samples)):
#         params = get_params()
#         params2 = get_params()  # np.array([0.25,0.25,0.25,0.25])
#         diff = np.expand_dims(np.expand_dims(np.squeeze(np.array(utils.beerlamb_multi(molecules, x, params, config.left_cut)))
#                                              / np.squeeze(np.array(utils.beerlamb_multi(molecules, x, params2, config.left_cut))), axis=1), axis=0)
#         if i == 0:
#             syn_data = diff
#             syn_params = np.expand_dims(params, axis=0) - np.expand_dims(params2, axis=0)
#         else:
#             syn_data = np.concatenate([syn_data, diff])
#             syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0) - np.expand_dims(params2, axis=0)])
            
#     print(syn_data.shape)
#     save_data(syn_data, syn_params)

def generate_delta_attenuation_dataset(n_samples=10000):
    wavelengths = scipy.io.loadmat(config.dataset_path + 'LWP483_10Jan2017_SharedHyperProbe.mat')['wavelengths'].astype(float)
    molecules, x = prepro.read_molecules(config.left_cut, config.right_cut, wavelengths)

    y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _, y_cytoa_diff, _, _, _, _ = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f)[:,0],
                                np.asarray(y_hb_f)[:,0],
                                np.asarray(y_coxa[:,0] - y_creda[:,0]))))
    
    syn_data, syn_params = np.zeros((n_samples, x.shape[0],1)), np.zeros((n_samples, config.molecule_count))
    for i in tqdm(range(n_samples)):
        params = get_no_scattering_params()
        delta_attenuation = beerlamb_delta_attenuation(M, params)
        params = params[:config.molecule_count]
        syn_data[i,:] = np.expand_dims(delta_attenuation,axis=1)
        syn_params[i,:] = np.expand_dims(params, axis=0)    
        
        # if i == 0:
        #     syn_data = delta_attenuation
        #     syn_params = np.expand_dims(params, axis=0)
        # else:
        #     syn_data = np.concatenate([syn_data, delta_attenuation])
        #     syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
            
    #print(syn_params.shape)
    save_data(syn_data, syn_params, False)   
    
def generate_scattering_delta_attenuation_dataset(n_samples=10000):
    wavelengths = scipy.io.loadmat(config.dataset_path + 'LWP483_10Jan2017_SharedHyperProbe.mat')['wavelengths'].astype(float)
    molecules, x = prepro.read_molecules(config.left_cut, config.right_cut, wavelengths)

    y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _, y_cytoa_diff, _, _, _, _ = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f)[:,0],
                                np.asarray(y_hb_f)[:,0],
                                np.asarray(y_coxa - y_creda)[:,0])))
    syn_data, syn_params = np.zeros((n_samples, x.shape[0],1)), np.zeros((n_samples, config.molecule_count))
    for i in tqdm(range(n_samples)):
        params = get_scattering_params()
        delta_attenuation = beerlamb_scattering_delta_attenuation(np.squeeze(x), M, params)
        params = params[:config.molecule_count]
        syn_data[i,:] = np.expand_dims(delta_attenuation,axis=1)
        syn_params[i,:] = np.expand_dims(params, axis=0)    
        
        # if i == 0:
        #     syn_data = delta_attenuation
        #     syn_params = np.expand_dims(params, axis=0)
        # else:
        #     syn_data = np.concatenate([syn_data, delta_attenuation])
        #     syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
            
    #print(syn_params.shape)
    save_data(syn_data, syn_params, True)   
   
def generate_delta_attenuation_dataset_inplace(n_samples=10000, left_cut=780, right_cut=900):
    wavelengths = scipy.io.loadmat(config.dataset_path + 'LWP483_10Jan2017_SharedHyperProbe.mat')['wavelengths'].astype(float)
    molecules, x = prepro.read_molecules(left_cut, right_cut, wavelengths)

    y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water, y_fat, y_cytoa_diff, _, _, _, _ = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f)[:,0],
                                np.asarray(y_hb_f)[:,0],
                                np.asarray(y_coxa[:,0] - y_creda[:,0]))))
                                #np.asarray(y_fat)[:,0])))
    
    syn_data, syn_params = np.zeros((n_samples, x.shape[0],1)), np.zeros((n_samples, config.molecule_count))
    for i in tqdm(range(n_samples)):
        params = get_no_scattering_params()
        delta_attenuation = beerlamb_delta_attenuation(M, params)
        params = params[:config.molecule_count]
        syn_data[i,:] = np.expand_dims(delta_attenuation,axis=1)
        syn_params[i,:] = np.expand_dims(params, axis=0)    
        
        # if i == 0:
        #     syn_data = delta_attenuation
        #     syn_params = np.expand_dims(params, axis=0)
        # else:
        #     syn_data = np.concatenate([syn_data, delta_attenuation])
        #     syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
            
    #print(syn_params.shape)
    return syn_data, syn_params, x, M
    
def generate_delta_attenuation_dataset_inplace_chromophores(chromophores, n_samples=10000, left_cut=780, right_cut=900):
    wavelengths = scipy.io.loadmat(config.dataset_path + 'LWP483_10Jan2017_SharedHyperProbe.mat')['wavelengths'].astype(float)
    molecules, x = prepro.read_molecules(left_cut, right_cut, wavelengths)

    #y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water, y_fat, y_cytoa_diff, _, _, _, _ = molecules
    #M = np.transpose(np.vstack((np.asarray(y_hbo2_f)[:,0],
    #                            np.asarray(y_hb_f)[:,0],
    #                            np.asarray(y_coxa[:,0] - y_creda[:,0]))))
                                #np.asarray(y_fat)[:,0])))
    M = np.transpose(np.vstack(chromophores))
    
    syn_data, syn_params = np.zeros((n_samples, x.shape[0],1)), np.zeros((n_samples, len(chromophores)))
    for i in tqdm(range(n_samples)):
        params = np.array(np.random.rand(len(chromophores)))-0.5
        delta_attenuation = beerlamb_delta_attenuation(M, params)
        params = params[:len(chromophores)]
        syn_data[i,:] = np.expand_dims(delta_attenuation,axis=1)
        syn_params[i,:] = np.expand_dims(params, axis=0)    
        
        # if i == 0:
        #     syn_data = delta_attenuation
        #     syn_params = np.expand_dims(params, axis=0)
        # else:
        #     syn_data = np.concatenate([syn_data, delta_attenuation])
        #     syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
            
    #print(syn_params.shape)
    return syn_data, syn_params, x, M

def generate_scattering_delta_attenuation_dataset_helicoid(minsmaxs, n_samples=10000):
    print("Helicoid: left_cut = ", left_cut, " right_cut = ", right_cut)
    hdr_path = path+"/{}/raw.hdr".format("020-01")
    img = open_image(hdr_path)
    wavelength = np.array(img.metadata['wavelength']).astype(float)
    molecules, x = read_molecules_creatis(left_cut_helicoid, right_cut_helicoid, x_waves=wavelength)
    y_hb_f, y_hbo2_f, y_coxa, y_creda, y_fat, y_water = molecules
    
    molecules_cyto, x_cyto = read_molecules_cytochrome_cb(left_cut_helicoid, right_cut_helicoid, wavelength)
    y_c_oxy, y_c_red, y_b_oxy, y_b_red = molecules_cyto #, y_water, y_fat = molecules

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
    
    syn_data, syn_params = np.zeros((n_samples, x.shape[0],1)), np.zeros((n_samples, M.shape[1]))
    for i in tqdm(range(n_samples)):
        params = get_scattering_params_helicoid(minsmaxs)
        delta_attenuation = beerlamb_scattering_delta_attenuation(np.squeeze(x), M, params)
        params = params[:M.shape[1]]
        syn_data[i,:] = np.expand_dims(delta_attenuation,axis=1)
        syn_params[i,:] = np.expand_dims(params, axis=0)        
    
    save_data_helicoid(syn_data, syn_params)
    


def generate_dataset_max_uniform(param_maxs, n_samples=10000):
    """
    generates synthetic dataset (spectrogram, param) pairs with n_samples.
    params are sampled from a uniform distribution with each parameter getting a max value from the array <param_maxs>
    """
    molecules, x = load_Xwaves()

    syn_data, syn_params = None, None
    for i in tqdm(range(n_samples)):
        params = np.array([np.random.uniform(0.0, m) for m in param_maxs])
        if i == 0:
            syn_data = np.expand_dims(utils.beerlamb_multi(molecules, x, params, config.left_cut), axis=0)
            syn_params = np.expand_dims(params, axis=0)
        else:
            syn_data = np.concatenate([syn_data, np.expand_dims(utils.beerlamb_multi(molecules, x, params, config.left_cut), axis=0)])
            syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
    save_data(syn_data, syn_params)


def generate_spectrogram(params=None, interpolate=False):
    """
    generates one synthetic (spectrogram, params) pair
    """
    molecules, x = load_Xwaves()

    if params is None:
        params = get_params()
    syn_data = np.expand_dims(utils.beerlamb_multi(molecules, x, params, config.left_cut), axis=0)
    syn_params = np.expand_dims(params, axis=0)
    return x, torch.from_numpy(syn_data).float(), torch.from_numpy(syn_params).float()


def generate_and_compare_custom(real_spectrogram, param_list):
    """
    not very smart "brute force" method for finding optimal parameters.
    probably wont be used. if interested see notebook -> unused_code/dict_method.ipynb
    """
    molecules, x = load_Xwaves()

    syn_params, best, best_params = None, None, None
    mse_min = 1000000
    mse_list = []

    for i, params in enumerate(tqdm(param_list)):
        if i == 0:
            syn_spectrogram = utils.beerlamb_multi(molecules, x, params, config.left_cut)
            syn_params = np.expand_dims(params, axis=0)
        else:
            syn_spectrogram = utils.beerlamb_multi(molecules, x, params, config.left_cut)
            syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
        syn_spectrogram = np.array(syn_spectrogram)
        syn_spectrogram /= np.max(syn_spectrogram)
        mse_new = mean_squared_error(real_spectrogram, syn_spectrogram)
        mse_list.append(mse_new)
        if mse_new < mse_min:
            best = syn_spectrogram
            best_params = params
            mse_min = mse_new

    return best, best_params, mse_min, real_spectrogram

# uncomment the next line to generate a new dataset
# generate_diff_dataset()
