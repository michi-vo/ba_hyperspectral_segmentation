import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import medfilt, savgol_filter, wiener
import os
from matplotlib.colors import SymLogNorm
import bisect
from spectral import * 
import random
from tqdm.auto import tqdm
from scipy.linalg import pinv
from PIL import Image
import matplotlib.cm as cm
import time
import config
from config import path_absorp, path_creatis, path_helicoid, left_cut_helicoid, right_cut_helicoid
import matplotlib as mpl
import imageio as io
import scipy


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


def beerlamb_multi(molecules, wavelength_range, a_all, left_cut):
    """
        Computes Beer-Lambert Law for multiple molecules.
        molecules: molecule absorption spectra. use the output of preprocessing.read_molecules() here
        wavelength_range: available wavelengths
        a_all: parameters (molecule concentrations)
        left_cut: left side cutoff of wavelengths
    """
    y = []
    for i in wavelength_range:
        b_avg = 0.86
        g_avg = 0.9
        m_a = 0
        step = int(i - left_cut)

        for j in range(len(a_all)):
            m_a += a_all[j] * molecules[j][step]

        mu_s_avg = np.power(np.divide(i, 500), -b_avg)
        mu_s_avg = mu_s_avg / (1 - g_avg)

        exp_m = np.exp(-(m_a))  # + mu_s_avg

        sp = exp_m ** 2  # signal Power
        snr = 8000  # signal to noise ratio
        std_n = (sp / snr) ** 0.5  # noise std. deviation

        y.append(exp_m + np.random.normal(0, std_n, 1)[0])

    return y  # I = I_o * exp(-(m_a+m_s)x)


def beerlamb_multi_batch(molecules, wavelength_range, a_all, left_cut):
    """
    beerlamb_multi but with batched input
    """
    y = np.empty((len(wavelength_range), a_all.shape[1]))
    for idx, i in enumerate(wavelength_range):
        b_avg = 0.86
        g_avg = 0.9
        m_a = 0
        step = int(i - left_cut)

        for j in range(len(a_all)):
            m_a += a_all[j] * molecules[j][step]

        mu_s_avg = np.power(np.divide(i, 500), -b_avg)
        mu_s_avg = mu_s_avg / (1 - g_avg)

        exp_m = np.exp(-(m_a))  # + mu_s_avg

        sp = exp_m ** 2  # signal Power
        snr = 8000  # signal to noise ratio
        std_n = (sp / snr) ** 0.5  # noise std. deviation

        y[idx] = exp_m + np.random.normal(0, std_n, exp_m.shape)

    return y  # I = I_o * exp(-(m_a+m_s)x)

# def beerlamb_scattering_delta_attenuation(wavelengths, M, params): #delta_c_i, a_t1, b_t1, b_ti, a_ti): #Input: molecules, wavelengths (already cut in correct range!) and as np.array
#     delta_c_i, a_t1, b_t1, a_ti, b_ti = params[:3], params[3], params[4], params[5], params[6]
#     S = ((((wavelengths/500)**(-b_ti)) * a_ti) - (((wavelengths/500)**(-b_t1)) * a_t1)) / (1-0.9)
#     delta_attenuation = (M @ delta_c_i) + S
#     return delta_attenuation

def beerlamb_scattering_delta_attenuation(wavelengths, M, params): #delta_c_i, a_t1, b_t1, b_ti, a_ti): #Input: molecules, wavelengths (already cut in correct range!) and as np.array
    delta_c_i, a_t1, b_t1, a_ti, b_ti = params[:M.shape[1]], params[M.shape[1]], params[M.shape[1]+1], params[M.shape[1]+2], params[M.shape[1]+3]
    S = ((((wavelengths/500)**(-b_ti)) * a_ti) - (((wavelengths/500)**(-b_t1)) * a_t1)) / (1-0.9)
    delta_attenuation = (M @ delta_c_i) + S
    return delta_attenuation

def beerlamb_delta_attenuation(M, params): #delta_c_i, a_t1, b_t1, b_ti, a_ti): #Input: molecules, wavelengths (already cut in correct range!) and as np.array
    delta_c_i = params
    delta_attenuation = (M @ delta_c_i)
    return delta_attenuation
    
def get_mean_std(dataset):
    return torch.mean(dataset), torch.std(dataset)


def plot_pred(opti_coef, nn_coef, name):
    """
    Used during training to plot prediction results
    """
    opti_coef = opti_coef.cpu().detach().numpy()
    nn_coef = nn_coef.cpu().detach().numpy()

    torch.save(opti_coef, 'results_piglets/{}/coef_Opti.pt'.format(name))
    torch.save(nn_coef, 'results_piglets/{}/coef_NN.pt'.format(name))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    coef = ["HbO2", "Hbb", "diffCCO"]
    #print(opti_coef.shape)
    #print(nn_coef.shape)
    
    for i in range(opti_coef.shape[1]):
        plt.plot(opti_coef[:, i], color=colors[i], label=f'Opti ' + coef[i], linewidth=3, alpha=0.5)
        plt.plot(nn_coef[:, i], color=colors[i], label=f'NN ' + coef[i], linewidth=0.5)

    plt.legend()
    plt.savefig("results_piglets/{}/fig.png".format(name), format='png')
    plt.clf()
    
def plot_helicoid_pred(opti_coef, nn_coef, name, img_sizes, patient='default'):
    """
    Used during training to plot prediction results
    """
    img_sizes = np.array(img_sizes)
    
    #print(img_sizes)
    assert len(img_sizes) == 1
    
    opti_coef = opti_coef.reshape(img_sizes[0][0], img_sizes[0][1], opti_coef.shape[-1])
    nn_coef = nn_coef.reshape(img_sizes[0][0], img_sizes[0][1], nn_coef.shape[-1])
    
    torch.save(opti_coef, 'results/{}/coef_Opti.pt'.format(name))
    torch.save(nn_coef, 'results/{}/coef_NN.pt'.format(name))

    # fig, (ax0,ax1,ax2,ax3) = plt.subplots(1,4,figsize=(24,8))
    
    # opti_hbt = opti_coef[:,:,0] + opti_coef[:,:,1]
    # nn_hbt = nn_coef[:,:,0] + nn_coef[:,:,1]
    
    # hbt_min = np.min([opti_hbt, nn_hbt])
    # hbt_max = np.max([opti_hbt, nn_hbt])
    # opti_hbt_img = ax0.imshow(opti_hbt, cmap='Reds', vmin=hbt_min, vmax=hbt_max)
    # ax0.set_title("Optimization HbT")
    # nn_hbt_img = ax1.imshow(nn_hbt, cmap='Reds', vmin=hbt_min, vmax=hbt_max)
    # ax1.set_title("NN HbT")
    
    # opti_diffCCO = opti_coef[:,:,2] - opti_coef[:,:,3]
    # nn_diffCCO = nn_coef[:,:,2] - nn_coef[:,:,3]
    # diffCCO_min = np.min([opti_diffCCO, nn_diffCCO])
    # diffCCO_max = np.max([opti_diffCCO, nn_diffCCO])
    
    # opti_diffCCO_img = ax2.imshow(opti_diffCCO, cmap='terrain', vmin=diffCCO_min, vmax=diffCCO_max)
    # ax2.set_title("Optimization diffCCO")
    # nn_diffCCO_img = ax3.imshow(nn_diffCCO, cmap='terrain', vmin=diffCCO_min, vmax=diffCCO_max)
    # ax3.set_title("NN diffCCO")

    # #fig.colorbar(opti_hbt_img, ax=ax0)
    # #fig.colorbar(nn_hbt_img, ax=ax1)
    # #fig.colorbar(opti_diffCCO_img, ax=ax2)
    # #fig.colorbar(nn_diffCCO_img, ax=ax3)
    
    # #plt.legend()
    # fig.savefig("results/{}/{}.png".format(name, patient), format='png')
    # plt.close(fig)
    # plt.clf()
    
    # opti_coef = opti_coef.cpu().detach().numpy()
    # nn_coef = nn_coef.cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))
    
    opti_hbt = opti_coef[:,:,0] + opti_coef[:,:,1]
    nn_hbt = nn_coef[:,:,0] + nn_coef[:,:,1]
    hbt_min = np.min([opti_hbt, nn_hbt])
    hbt_max = np.max([opti_hbt, nn_hbt])
    opti_diffCCO = opti_coef[:,:,2] - opti_coef[:,:,3]
    nn_diffCCO = nn_coef[:,:,2] - nn_coef[:,:,3]
    diffCCO_min = np.min([opti_diffCCO, nn_diffCCO])
    diffCCO_max = np.max([opti_diffCCO, nn_diffCCO])
    
    opti_hbt_img = axes[0].imshow(opti_hbt, cmap='Reds', vmin=hbt_min, vmax=hbt_max)
    axes[0].set_title("Optimization HbT", fontsize=15)
    nn_hbt_img = axes[1].imshow(nn_hbt, cmap='Reds', vmin=hbt_min, vmax=hbt_max)
    axes[1].set_title("NN HbT", fontsize=15)

    cb_hbt = fig.colorbar(opti_hbt_img, ax=axes[0:2], orientation='horizontal', shrink=0.7, pad=0.1)
    cb_hbt.set_label('HbT Scale', fontsize=15)

    #h_scale = 0.000003
    opti_diffCCO_img = axes[2].imshow(opti_diffCCO, cmap='terrain', vmin=diffCCO_min, vmax=diffCCO_max)
    
    #opti_diffCCO_img = axes[2].imshow(opti_diffCCO, cmap='terrain', vmin=0.01, vmax=diffCCO_max/2)
    #opti_diffCCO_img = axes[2].imshow(opti_diffCCO, cmap='terrain', norm=SymLogNorm(linthresh=h_scale, linscale=h_scale, vmin=diffCCO_min, vmax=diffCCO_max))
    axes[2].set_title("Optimization diffCCO", fontsize=15)
    nn_diffCCO_img = axes[3].imshow(nn_diffCCO, cmap='terrain', vmin=diffCCO_min, vmax=diffCCO_max)
    
    #nn_diffCCO_img = axes[3].imshow(nn_diffCCO, cmap='terrain', vmin=0.01, vmax=diffCCO_max/2)
    #nn_diffCCO_img = axes[3].imshow(nn_diffCCO, cmap='terrain', norm=SymLogNorm(linthresh=h_scale, linscale=h_scale, vmin=diffCCO_min, vmax=diffCCO_max))
    axes[3].set_title("NN diffCCO", fontsize=15)

    cb_diffCCO = fig.colorbar(opti_diffCCO_img, ax=axes[2:4], orientation='horizontal', shrink=0.7, pad=0.1)
    cb_diffCCO.set_label('diffCCO Scale', fontsize=15)

    # Display the plot
    fig.savefig("results/{}/{}.png".format(name, patient), format='png')
    plt.close(fig)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    hbt_abs_diff = np.abs(opti_hbt - nn_hbt)
    diffcco_abs_diff = np.abs(opti_diffCCO - nn_diffCCO)

    # Find min and max across both images for consistent colorbar
    hbt_diff_min = np.min(hbt_abs_diff)
    hbt_diff_max = np.max(hbt_abs_diff)
    diffcco_diff_min = np.min(diffcco_abs_diff)
    diffcco_diff_max = np.max(diffcco_abs_diff)

    # Display the absolute differences
    hbt_diff_img = axes[0].imshow(hbt_abs_diff, cmap='Reds', vmin=hbt_diff_min, vmax=hbt_diff_max)
    axes[0].set_title("HbT Absolute Difference", fontsize=15)
    diffcco_diff_img = axes[1].imshow(diffcco_abs_diff, cmap='terrain', vmin=diffcco_diff_min, vmax=diffcco_diff_max)
    axes[1].set_title("diffCCO Absolute Difference", fontsize=15)

    # Add colorbars to the right of the images
    cb_hbt_diff = fig.colorbar(hbt_diff_img, ax=axes[0], orientation='vertical')
    cb_hbt_diff.set_label('HbT Difference Scale', fontsize=15)
    cb_diffcco_diff = fig.colorbar(diffcco_diff_img, ax=axes[1], orientation='vertical')
    cb_diffcco_diff.set_label('diffCCO Difference Scale', fontsize=15)

    # Save the plot with a different filename
    fig.savefig("results/{}/{}_diff.png".format(name, patient), format='png')
    plt.close(fig)

    plt.clf()
    
def plot_helicoid_pred_no_scatter(path_helicoid, patient):
    hdr_path = path_helicoid+"/{}/raw.hdr".format(patient)
    start_time = time.time()
    img = open_image(hdr_path)
    end_time = time.time()
    print("Time taken to open image: {} seconds".format(end_time - start_time))
    wavelength = np.array(img.metadata['wavelength']).astype(float)
    #print(wavelength)

    rbg_path = path_helicoid+"/{}/image.jpg".format(patient)
    rbg = Image.open(rbg_path)

    gt_path = path_helicoid+"/{}/gtMap.hdr".format(patient)
    gt = open_image(gt_path)
    gt = gt.load()

    white_path = path_helicoid+"/{}/whiteReference.hdr".format(patient)
    white = open_image(white_path)
    white = white.load()

    dark_path = path_helicoid+"/{}/darkReference.hdr".format(patient)

    dark = open_image(dark_path)
    dark = dark.load()
    #white_full = np.tile(white, (img.shape[0],1,1))
    #white_full_RGB = np.stack((white_full[:,:,int(img.metadata['default bands'][2])],white_full[:,:,int(img.metadata['default bands'][1])],white_full[:,:,int(img.metadata['default bands'][0])]), axis=2)
    #dark_full = np.tile(dark, (img.shape[0],1,1))
    white_full = np.broadcast_to(np.array(white), img.shape)
    # white_full_RGB = np.stack((white_full[:,:,int(img.metadata['default bands'][2])],white_full[:,:,int(img.metadata['default bands'][1])],white_full[:,:,int(img.metadata['default bands'][0])]), axis=2)
    dark_full = np.broadcast_to(np.array(dark), img.shape) / 2

    #img_normalized = np.array(img.load())
    #img_normalized = (img.load() - dark_full) / (white_full - dark_full)
    #img_normalized = img_normalized * 2
    img_normalized = (img.load() - dark_full) / (white_full - dark_full)
    img_normalized[img_normalized <= 0] = 10**-2
    
    img_RGB = np.stack((img_normalized[:,:,int(img.metadata['default bands'][2])],img_normalized[:,:,int(img.metadata['default bands'][1])],img_normalized[:,:,int(img.metadata['default bands'][0])]), axis=2)

    tumor = gt==2
    normal = gt==1
    blood = gt==3
    mask1 = np.ma.masked_where(tumor.astype(int) == 0, tumor.astype(int))
    mask2 = np.ma.masked_where(normal.astype(int) == 0, normal.astype(int))
    mask3 = np.ma.masked_where(blood.astype(int) == 0, blood.astype(int))

    molecules, x = read_molecules_creatis(left_cut_helicoid, right_cut_helicoid, x_waves=wavelength)
    y_hb_f, y_hbo2_f, y_coxa, y_creda, y_fat, y_water = molecules

    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa),
                                np.asarray(y_creda),
                                np.asarray(y_water),
                                np.asarray(y_fat))))

    M_pinv = pinv(M)

    (x_normal,y_normal) = np.where((np.squeeze(gt)==1))
    (x_tumor,y_tumor) = np.where((np.squeeze(gt)==2))
    (x_blood,y_blood) =  np.where((np.squeeze(gt)==3))

    hbt_inferred = np.zeros((img.shape[0], img.shape[1]))
    hbdiff_inferred = np.zeros((img.shape[0], img.shape[1]))
    error_inferred = np.zeros((img.shape[0], img.shape[1]))
    blood_scattering_inferred = np.zeros((img.shape[0], img.shape[1]))
    diffCCO_inferred = np.zeros((img.shape[0], img.shape[1]))
    CCO_inferred = np.zeros((img.shape[0], img.shape[1]))
    brain_scattering_inferred = np.zeros((img.shape[0], img.shape[1]))
    tumor_scattering_inferred = np.zeros((img.shape[0], img.shape[1]))

    #reference_index_normal = random.randint(0, len(x_normal) - 1)
    #reference_point = [x_normal[reference_index_normal], y_normal[reference_index_normal]]
    reference_index_normal = random.randint(0, len(x_blood) - 1)
    reference_point = [x_blood[reference_index_normal], y_blood[reference_index_normal]]

    reference = (img_normalized[reference_point[0], reference_point[1], :])


    for i in tqdm(range(0, img.shape[0])):
        for j in range(0, img.shape[1]):
            orange_dot = [i,j]
            #blue_dot = closest_normal(orange_dot)
            #relative_attenuation = (-np.log(img_normalized[orange_dot[0], orange_dot[1], :] / img_normalized[blue_dot[0], blue_dot[1], :]))[np.in1d(wavelength,x)]
            
            #relative_attenuation = (-np.log(((img_normalized[orange_dot[0], orange_dot[1], :]) / (mean_normal))))[np.in1d(wavelength,x)]
            relative_attenuation = (-np.log(((img_normalized[orange_dot[0], orange_dot[1], :]) 
                                            / reference)))[np.in1d(wavelength,x)]
            
            concentrations = (M_pinv @ relative_attenuation)
            error_inferred[i,j] = np.mean(np.abs((M @ concentrations) - relative_attenuation))
            hbt_inferred[i,j] = concentrations[0] + concentrations[1]
            hbdiff_inferred[i,j] = concentrations[1] - concentrations[0]
            diffCCO_inferred[i,j] = concentrations[2] - concentrations[3]
            CCO_inferred[i,j] = concentrations[2] + concentrations[3]
    
    from matplotlib.colors import Normalize
    error_tolerance = 0.5

    (x_normal,y_normal) = np.where((np.squeeze(gt)==1) & (error_inferred < error_tolerance))
    (x_tumor,y_tumor) = np.where((np.squeeze(gt)==2) & (error_inferred < error_tolerance))
    (x_blood,y_blood) =  np.where((np.squeeze(gt)==3) & (error_inferred < error_tolerance))

    fig, axs = plt.subplots(nrows=2, ncols=3,figsize=(16,10))
    ax_RGB, ax_hbt, ax_diffcco, ax_greenpoint, ax_bluepoint, ax_yellowpoint = axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]
    ax_RGB.imshow(img_RGB)
    ax_RGB.set_title("Synthetic RGB", fontsize=20)
    ax_RGB.imshow(mask1, cmap=cm.tab10)
    ax_RGB.imshow(mask2, cmap=cm.Dark2)
    ax_RGB.imshow(mask3, cmap=cm.get_cmap('spring_r'))

    hbt_filtered = hbt_inferred
    hbt_filtered = np.where(error_inferred > error_tolerance, np.nan, hbt_filtered)
    hbt_img = ax_hbt.imshow(hbt_filtered, cmap='Reds')
    fig.colorbar(hbt_img, ax=ax_hbt)
    ax_hbt.set_title("Inferred HbT > 0 \n Fitting MAE < " + str(error_tolerance), fontsize=15)

    diffCCO_inferred_filtered = np.where(error_inferred > error_tolerance, np.nan, diffCCO_inferred)
    #diffCCO_inferred_filtered = np.where(diffCCO_inferred_filtered > 0, 0, diffCCO_inferred_filtered)
    diffCCO_img = ax_diffcco.imshow(diffCCO_inferred_filtered)
    fig.colorbar(diffCCO_img, ax=ax_diffcco)
    ax_diffcco.set_title("diffCCO (MAE < " + str(error_tolerance) +")")

    rows, cols = np.indices(error_inferred.shape)
    green_index = random.randint(0, len(x_normal) - 1)
    green_point = [x_normal[green_index], y_normal[green_index]]

    yellow_index = random.randint(0, len(x_blood) - 1)
    yellow_point = [x_blood[yellow_index], y_blood[yellow_index]]

    ax_RGB.plot(green_point[1], green_point[0], marker="o", markersize=10, markeredgecolor="darkgreen", markerfacecolor='#66FF66')
    ax_RGB.plot(yellow_point[1], yellow_point[0], marker="o", markersize=10, markeredgecolor="yellow", markerfacecolor="yellow")

    (x_normal,y_normal) = np.where((np.squeeze(gt)==1) & (error_inferred < error_tolerance))
    (x_tumor,y_tumor) = np.where((np.squeeze(gt)==2) & (error_inferred < error_tolerance))
    (x_blood,y_blood) =  np.where((np.squeeze(gt)==3) & (error_inferred < error_tolerance))

    ax_RGB.plot(reference_point[1], reference_point[0], marker="x", markersize=10, markeredgecolor="orange", markerfacecolor="#FFA500")


    relative_attenuation = (-np.log(((img_normalized[green_point[0], green_point[1], :]) / reference))[np.in1d(wavelength,x)])
    inferred_attenuation = M @ (M_pinv @ relative_attenuation)
    ax_greenpoint.plot(x, relative_attenuation, label='GT')
    ax_greenpoint.plot(x, inferred_attenuation,  alpha=0.8, linewidth=4, label='BLL')
    ax_greenpoint.legend()
    #print(error_inferred[green_point[0], green_point[1]])
    #print(np.mean(np.abs((inferred_attenuation - relative_attenuation))))
    assert error_inferred[green_point[0], green_point[1]] == np.mean(np.abs((inferred_attenuation - relative_attenuation)))
    ax_greenpoint.set_title("Green point with MAE " + str(round(error_inferred[green_point[0], green_point[1]],3)) + " (Normal)", fontsize=12)

    if len(x_tumor) > 0:
        blue_index = random.randint(0, len(x_tumor) - 1)
        blue_point = [x_tumor[blue_index], y_tumor[blue_index]]
        ax_RGB.plot(blue_point[1], blue_point[0], marker="o", markersize=10, markeredgecolor="darkblue", markerfacecolor="#0000FF")
        relative_attenuation = (-np.log((img_normalized[blue_point[0], blue_point[1], :]) / reference))[np.in1d(wavelength,x)]
        inferred_attenuation = M @ (M_pinv @ relative_attenuation)
        ax_bluepoint.plot(x, relative_attenuation, label='GT')
        ax_bluepoint.plot(x, inferred_attenuation, alpha=0.8, linewidth=4, label='BLL')
        ax_bluepoint.legend()
        assert error_inferred[blue_point[0], blue_point[1]] == np.mean(np.abs((inferred_attenuation - relative_attenuation)))
        print(error_inferred[blue_point[0], blue_point[1]])
        ax_bluepoint.set_title("Blue point with MAE " + str(round(error_inferred[blue_point[0], blue_point[1]],3)) + " (Tumor)", fontsize=12)

    relative_attenuation = (-np.log((img_normalized[yellow_point[0], yellow_point[1], :]) / reference))[np.in1d(wavelength,x)]
    inferred_attenuation = M @ (M_pinv @ relative_attenuation)
    ax_yellowpoint.plot(x, relative_attenuation, label='GT')
    ax_yellowpoint.plot(x, inferred_attenuation, alpha=0.8, linewidth=4, label='BLL')
    ax_yellowpoint.legend()
    assert error_inferred[yellow_point[0], yellow_point[1]] == np.mean(np.abs((inferred_attenuation - relative_attenuation)))
    print(error_inferred[yellow_point[0], yellow_point[1]])
    ax_yellowpoint.set_title("Yellow point with MAE " + str(round(error_inferred[yellow_point[0], yellow_point[1]],3)) + " (Blood)", fontsize=12)

    #plt.savefig(plot_dir + "helicoid")
    plt.show()


def median_fil(data, window_size=17):
    window_size = 17  # Adjust as needed
    return medfilt(data, kernel_size=window_size)


def moving_avg_fil(data, window_size=17):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def savgol_fil(data, window_size=17, polyorder=1):
    return savgol_filter(data, window_length=window_size, polyorder=polyorder)


def wiener_fil(data, window_size=17):
    return wiener(data, mysize=window_size)


def save_optimization_data(ref_spectr, spectra_list, coef_list, folder, scattering_coef_list=None, t1=None, time_taken=None):
    """
    Used after running the optimisation script/notebook to be used later by the Neural Network
    """
    spectra_list = torch.swapaxes(torch.squeeze(torch.from_numpy(np.array(spectra_list))), 0, 1)
    coef_list = torch.swapaxes(torch.squeeze(torch.from_numpy(np.array(coef_list))), 0, 1)
    
    if scattering_coef_list is not None: scattering_coef_list = torch.swapaxes(torch.squeeze(torch.from_numpy(np.array(scattering_coef_list))), 0, 1)
    if t1 is not None: t1 = torch.from_numpy(np.array(t1))
    if time_taken is not None: time_taken = torch.from_numpy(np.array([time_taken]))
    
    if config.scattering_model:
        print("Saving scattering opti data")
        path = config.dataset_path + "piglet_scattering/"
    else:
        print("Saving no scattering opti data")
        path = config.dataset_path + "piglet_no_scattering/"
        
    if not os.path.exists(path): os.mkdir(path)
    path = path + folder 
    
    #path = config.dataset_path + 'piglet_diffs/' + folder
    if not os.path.exists(path): os.mkdir(path)
    
    print("Saving in ", path)
    torch.save(ref_spectr, path + '/ref_spectr.pt')
    torch.save(spectra_list, path + '/spectra_list.pt')
    torch.save(coef_list, path + '/coef_list.pt')
    if scattering_coef_list != None: torch.save(scattering_coef_list, path + '/scattering_coef_list.pt')
    if t1 != None: torch.save(t1, path + '/t1.pt')
    if time_taken != None: torch.save(time_taken, path + '/time_taken.pt')


def save_helicoid_optimization_data(img_ref, reference_point, coef_list, scattering_params, errors, t1, time_taken, folder):
    """
    Used after running the optimisation script/notebook to be used later by the Neural Network
    """
    img_ref = torch.from_numpy(img_ref)
    reference_point = torch.from_numpy(reference_point)
    coef_list = torch.from_numpy(coef_list)
    scattering_params = torch.from_numpy(scattering_params)
    errors = torch.from_numpy(errors)
    t1 = torch.from_numpy(t1)
    time_taken = torch.from_numpy(np.array([time_taken]))
    
    path = config.dataset_path + "helicoid/"
    
    if not os.path.exists(path): os.mkdir(path)
    path = path + folder 
    if not os.path.exists(path): os.mkdir(path)
    
    torch.save(img_ref, path+'/img_ref.pt')
    torch.save(reference_point, path+'/reference_point.pt')
    torch.save(coef_list, path + '/coef_list.pt')
    torch.save(scattering_params, path + '/scattering_params.pt')
    torch.save(errors, path + '/errors.pt')
    torch.save(t1, path + '/t1.pt')
    torch.save(time_taken, path + '/time_taken.pt')
    
def load_helicoid_optimization_data(folder):
    """
    Used to load the data saved by the save_helicoid_optimization_data function
    """
    path = config.dataset_path + "helicoid/" + folder

    img_ref = torch.load(path+'/img_ref.pt')
    reference_point = torch.load(path+'/reference_point.pt')
    coef_list = torch.load(path + '/coef_list.pt')
    scattering_params = torch.load(path + '/scattering_params.pt')
    errors = torch.load(path + '/errors.pt')
    t1 = torch.load(path + '/t1.pt')

    return img_ref, reference_point, coef_list, scattering_params, errors, t1

def load_helicoid_trainset(coarseness=1, calc_attenuation=False, add_constant=False):
    path = "/home/home/ivan/HELICoiD/HSI_Human_Brain_Database_IEEE_Access"
    dataset_path = "./dataset/"
    path_absorp = "./dataset/UCL-NIR-Spectra/spectra/"
    path_creatis = "./dataset/CREATIS-Spectra/spectra/"
    
    all_imgs = None
    all_gts = None
    
    for patient in tqdm(["008-01", "008-02", "016-04", "016-05", "020-01", "025-02"]):
        hdr_path = path+"/{}/raw.hdr".format(patient)
        img = open_image(hdr_path)
        wavelength = np.array(img.metadata['wavelength']).astype(float)        
        rbg_path = path+"/{}/image.jpg".format(patient)
        rbg = Image.open(rbg_path)

        gt_path = path+"/{}/gtMap.hdr".format(patient)
        gt = open_image(gt_path)
        gt = gt.load()
        
        white_path = path+"/{}/whiteReference.hdr".format(patient)
        white = open_image(white_path)
        white = white.load()

        dark_path = path+"/{}/darkReference.hdr".format(patient)

        dark = open_image(dark_path)
        dark = dark.load()
        white_full = np.tile(white, (img.shape[0],1,1))
        dark_full = np.tile(dark, (img.shape[0],1,1))
        img_normalized = ((img.load() - dark_full) / (white_full - dark_full))
        img_normalized[img_normalized <= 0] = 10**-3
        
        if add_constant:
            img_normalized = img_normalized + 0.1
        
        img_normalized = img_normalized[::coarseness, ::coarseness, :]
        gt = gt[::coarseness, ::coarseness, :]
        
        normal = gt == 1
        tumor = gt == 2
        blood = gt == 3
        x_blood,y_blood,z_blood = np.where(blood==1)
        x_tumor,y_tumor,z_tumor = np.where(tumor==1)
        x_normal,y_normal,z_normal = np.where(normal==1)
        
        if calc_attenuation:
            random_blood_x_index = random.randint(0, len(x_blood) - 1)
            random_blood_y_index = random.randint(0, len(y_blood) - 1)
            img_normalized = -np.log(img_normalized / (img_normalized[x_blood[random_blood_x_index], y_blood[random_blood_y_index],:]))
        
        if all_imgs is None:
            all_imgs = img_normalized
            all_gts = gt
        else:
            all_imgs = concatenate_images(all_imgs, img_normalized)
            all_gts = concatenate_images(all_gts, gt)
            
    return all_imgs, all_gts
        
def concatenate_images(img1, img2):
    # Determine the size of the images
    A, B, C = img1.shape
    D, E, _ = img2.shape

    # Pad the shorter image with zeros
    if A < D:
        # If the first image is shorter, pad it
        padding = np.zeros((D - A, B, C), dtype=img1.dtype)
        padded_img1 = np.concatenate((img1, padding), axis=0)
        result = np.concatenate((padded_img1, img2), axis=1)
    elif D < A:
        # If the second image is shorter, pad it
        padding = np.zeros((A - D, E, C), dtype=img2.dtype)
        padded_img2 = np.concatenate((img2, padding), axis=0)
        result = np.concatenate((img1, padded_img2), axis=1)
    else:
        # If both images have the same height, concatenate directly
        result = np.concatenate((img1, img2), axis=1)

    return result