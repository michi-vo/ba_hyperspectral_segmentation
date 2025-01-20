#!/usr/bin/env python
# coding: utf-8

# In[1]:


from spectral import * 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bisect
import imageio as io
from PIL import Image
import numpy as np
import scipy

mpl.rc('image', cmap='terrain')
mpl.rcParams['text.color'] = 'grey'
mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['ytick.color'] = 'grey'
mpl.rcParams['axes.labelcolor'] = 'grey'
#plt.rcParams["font.family"] = "Laksaman"

def remove_ticks(ax):
    ax.set_xticks([]) 
    ax.set_yticks([]) 


# In[80]:


path_absorp = "/home/kevinscibilia/UCL-NIR-Spectra/spectra/"
hbo2_absorp = path_absorp + "hb02.txt"
hhb_absorp = path_absorp + "hb.txt"
water_absorp = path_absorp + "matcher94_nir_water_37.txt"
diff_cyto_absorp = path_absorp + "cytoxidase_diff_odmMcm.txt"
fat_absorp = path_absorp + "fat.txt"
cyto_oxy_absorp = path_absorp + "moody cyt aa3 oxidised.txt"
cyto_red_absorp = path_absorp + "moody cyt aa3 reduced.txt"
### Using the preprocessing function we know
def load_spectra():
    path_absorp = "/home/kevinscibilia/UCL-NIR-Spectra/spectra/"
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
        #print("new_x", new_x)
        #print("x", x)
        #print("y[i]", y[i])
        print(i)
        new_y[i] = np.interp(new_x, x, y[i])

    return new_y, new_x

def read_molecules(left_cut, right_cut, x_waves=None):
    #print(x_waves)
    path_dict = load_spectra()

    # read spectra for: cytochrome oxydised/reduced, oxyhemoglobin, hemoglobin, water, fat
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600", "water", "fat" ,"cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600", "water", "fat"]
    
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600"]
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600","cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
    #mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600","cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red", "fat"]
    mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600","cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red", "fat", "water"]

    
    x, y = {}, {}
    for i in mol_list:
        x[i], y[i] = read_spectra(path_dict[i])

    # from extinction to absorption
    # TODO check if water spectra was in extinction 
    
    #y_list = ['hb_450', 'hb_600', 'hbo2_450', 'hbo2_600', 'cytoa_oxy', 'cytoa_red']
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
    
    # interpolate till 800nm as the data have value for every second nm, 400,402,404...
    # print("before")
    # print(x["hbo2"])
    # print(y["hbo2"])
    # for i in ["hbo2", "hb"]:
    #     xvals = np.array([i for i in range(int(x[i + "_450"][0]), int(x[i + "_600"][-1]) + 1)])
    #     yinterp = np.interp(xvals, np.concatenate([x[i + "_450"], x[i + "_600"]]), np.concatenate([y[i + "_450"], y[i + "_600"]]))
    #     x[i] = np.concatenate([xvals, x[i][151:]])
    #     y[i] = np.concatenate((yinterp, np.asarray(y[i][151:])), axis=None)
    # print("after")
    # print(x["hbo2"])
    # print(y["hbo2"])
    
    # cutting all spectra to the range [left_cut, right_cut] nm
    x_new = x["cytoa_oxy"][bisect.bisect_left(x["cytoa_oxy"], left_cut):bisect.bisect_right(x["cytoa_oxy"], right_cut)]
    #x_new = x["hbo2"][bisect.bisect_left(x["hbo2"], left_cut):bisect.bisect_right(x["hbo2"], right_cut)]

    
    #mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red", "water", "fat" ,"cytoa_diff", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
    #mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red", "water", "fat"]

    #mol_list = ["hbo2", "hb"]
    #mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red"]
    #mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red"]
    #mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red", "fat"]
    mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red", "cytoc_oxy", "cytoc_red", "cytob_oxy", "cytob_red", "fat", "water"]

    
    
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


# In[171]:


### There are 826 bands in the HELICoiD HSI cubes. We loop over and visualise them below
path = "/home/kevinscibilia/helicoid/HSI_Human_Brain_Database_IEEE_Access"

patients = ["017-01", "016-04", "016-05", "012-01", "020-01", "015-01", "012-02", "025-02", "021-01", "008-01", "008-02", "010-03", "021-05", "014-01"]

#patient = "020-01"
#patient = "015-01"
patient = "012-02"
#patient = "025-02"
#patient = "021-01"
#patient = "008-01"

#patient = "008-02"

#patient = "010-03"
#patient = "021-05"
#patient = "014-01"

#patient = "012-01"
#patient = "016-04"
#patient = "016-05"
#patient = "017-01"

hdr_path = path+"/{}/raw.hdr".format(patient)
img = open_image(hdr_path)
wavelength = np.array(img.metadata['wavelength']).astype(float)

for i in range(1,img.shape[-1]-50,50):

    
    fig, (ax, ax1, ax2, ax3) = plt.subplots(ncols=4,figsize=(15,4), 
            gridspec_kw={"width_ratios":[0.5,0.5,0.5,0.5]})
    fig.subplots_adjust(wspace=0.02)

    ax.imshow(img[:,:,i].squeeze())
    ax.set_title("Band #{}".format(i), fontsize=18)
    remove_ticks(ax)

    ax1.imshow(img[:,:,i+20].squeeze())
    ax1.set_title("Band #{}".format(i+20), fontsize=18)
    remove_ticks(ax1)

    ax2.imshow(img[:,:,i+30].squeeze())
    ax2.set_title("Band #{}".format(i+30), fontsize=18)
    remove_ticks(ax2)
    #ax2.imshow(mask2[:,:,110].T[10:220,15:225], cmap=cm.Dark2)

    ax3.imshow(img[:,:,i+40].squeeze())
    ax3.set_title("Band #{}".format(i+40), fontsize=18)
    remove_ticks(ax3)
    
    #if i==1:
    #    break


# In[172]:


#Check we sampled from the same wavelengths for all patients we consider
for patient_num in patients:
    hdr_path_i = path+"/{}/raw.hdr".format(patient_num)
    img_i = open_image(hdr_path_i)
    assert np.all(np.array(img_i.metadata['wavelength']).astype(float) == wavelength)


# In[173]:


### Visualising a synthetic RBG and label map
from matplotlib.colors import ListedColormap
## Image:
# The synthetic RGB image is generated by extracting three specific spectral bands from the HS cube with:
# red (708.97 nm), 
# green (539.44 nm), 
# and blue (479.06 nm) colors

## Labels:
# 0 = indicates pixels that are not labelled,
# 1 = Normal Tissue, 
# 2 = Tumour Tissue, 
# 3 = Hypervascularized Tissue, 
# 4 = Background

rbg_path = path+"/{}/image.jpg".format(patient)
rbg = Image.open(rbg_path)

gt_path = path+"/{}/gtMap.hdr".format(patient)
gt = open_image(gt_path)
gt = gt.load()
print(gt.shape)


fig, (ax, ax1) = plt.subplots(ncols=2,figsize=(10,10))

ax.imshow(rbg)
ax.set_title("Synthetic RBG Image", fontsize=18)

#ax1.imshow(rbg)
im = ax1.imshow(gt[:,:,0].squeeze(), cmap = ListedColormap(['white', 'green', 'red', 'blue', 'black']))
ax1.set_title("Labels", fontsize=18)
#plt.colorbar(im, ticks=[0, 1, 2, 3, 4], ax=ax1)
#np.unique(gt)

np.sum(gt[:,:,0] == 2)


# In[174]:


import matplotlib as mpl
mpl.rc('image', cmap='terrain')

mpl.rcParams['text.color'] = 'grey'
mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['ytick.color'] = 'grey'
mpl.rcParams['axes.labelcolor'] = 'grey'
#plt.rcParams["font.family"] = "Laksaman"

#

# #load
# #data_fk = nib.load(path + folder + "/inferred_tumor_patientspace_mri.nii")
# #data_fk = data_fk.get_fdata()
# #data_me = nib.load(path + folder + "/inferred_tumor_patientspace_m_mri_s.nii")
# #data_me = data_me.get_fdata()
# data_segm = nib.load(path + "/BraTS19_CBICA_BGG_1_seg.nii.gz")
# data_segm = data_segm.get_fdata()
tumor = gt==2
normal = gt==1
blood = gt==3

# t1gd_scan_dep = (data_segm==1) | (data_segm==4)
# enhanc_scan_dep1 = data_segm==1
# enhanc_scan_dep2 = data_segm==4
# enhanc_scan_dep2 = 2*enhanc_scaLinearSegmentedColormap.from_list('dark_red', [(0, '#000000'), (1, '#880000')], N=256)
#n_dep2

# data_brain = nib.load(path + "/BraTS19_CBICA_BGG_1_t1ce.nii.gz")
# data_brain = data_brain.get_fdata()
# data_brain = (data_brain - np.min(data_brain)) / (np.max(data_brain) - np.min(data_brain))

# #process
# mri_scan_dep = flair_scan_dep
mask1 = np.ma.masked_where(tumor.astype(int) == 0, tumor.astype(int))
mask2 = np.ma.masked_where(normal.astype(int) == 0, normal.astype(int))
mask3 = np.ma.masked_where(blood.astype(int) == 0, blood.astype(int))

#mask3 = np.ma.masked_where((enhanc_scan_dep1.astype(int) + enhanc_scan_dep2.astype(int)) == 0, (enhanc_scan_dep1.astype(int)+enhanc_scan_dep2.astype(int)))
#mask3 = np.ma.masked_where(enhanc_scan_dep2 == 0, enhanc_scan_dep2)



# #plot
# modelname_indep = "FK"
# modelname_dep = "ME"
# meanvalues = np.mean(mri_scan_dep, axis=(0,1))
# s = np.argmax(meanvalues)
# showparameters = 0

fig, (ax, ax1, ax2, ax3) = plt.subplots(ncols=4,figsize=(15,4), 
        gridspec_kw={"width_ratios":[0.5,0.5,0.5,0.5]})
fig.subplots_adjust(wspace=0.02)

ax.imshow(rbg) #, cmap=cm.gray)
ax.set_title("Synthetic RGB image", fontsize=18)
remove_ticks(ax)
#print(data_brain[:,:,s].T.shape)

ax1.imshow(rbg)
ax1.set_title("Tumor tissue", fontsize=18)
remove_ticks(ax1)
ax1.imshow(mask1, cmap=cm.tab10)

ax2.imshow(rbg)
ax2.set_title("Normal tissue", fontsize=18)
remove_ticks(ax2)
ax2.imshow(mask2, cmap=cm.Dark2)

ax3.imshow(rbg)
ax3.set_title("Blood vessels", fontsize=18)
remove_ticks(ax3)
ax3.imshow(mask3, cmap=cm.get_cmap('spring_r'))
plt.savefig("RGB_masks.pdf", dpi=200, bbox_inches="tight")


# In[175]:


x_blood,y_blood,z_blood = np.where(blood==1)

def closest_normal(index):
    i,j = index
    indices = np.argwhere(normal[:,:,0])
    distances = np.sqrt((indices[:, 0] - i) ** 2 + (indices[:, 1] - j) ** 2)
    min_dist_index = np.argmin(distances)
    return indices[min_dist_index]

# closest_normal([x_blood[0], y_blood[0]])


# In[176]:


### Visualising a synthetic RBG and label map

from scipy.signal import savgol_filter

def sfilter(x):
#    return medfilt(x)
    return savgol_filter(x,7,1)
    #return x

def shiftw(x):
    return np.roll(x,5)

## Image:
# The synthetic RGB image is generated by extracting three specific spectral bands from the HS cube with:
# red (708.97 nm), 
# green (539.44 nm), 
# and blue (479.06 nm) colors

## Labels:
# 0 = indicates pixels that are not labelled,
# 1 = Normal Tissue, 
# 2 = Tumour Tissue, 
# 3 = Hypervascularized Tissue, 
# 4 = Background

import os
#path = "/home/ivan/aimlab/HELICoiD/HSI_Human_Brain_Database_IEEE_Access"

#patient = "020-01"

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
white_full_RGB = np.stack((white_full[:,:,int(img.metadata['default bands'][2])],white_full[:,:,int(img.metadata['default bands'][1])],white_full[:,:,int(img.metadata['default bands'][0])]), axis=2)
dark_full = np.tile(dark, (img.shape[0],1,1)) / 2

#img_normalized = np.array(img.load())
img_normalized = (img.load() - dark_full) / (white_full - dark_full)
#img_normalized = img_normalized * 2
#img_normalized = sfilter((img.load() - dark_full) / (white_full - dark_full))
#img_normalized = shiftw(img_normalized)
#img_normalized = np.array(img.load())
img_normalized[img_normalized <= 0] = 10**-2
print(img_normalized.shape)
### Visualising spectrograms

#intensity1 = []
#intensity2 = []

#blue_dot = [200,100]
blood_index = 500
orange_dot = [x_blood[blood_index], y_blood[blood_index]]
blue_dot = closest_normal(orange_dot)

#blue_dot = [110,180]
#orange_dot = [220,160]
#orange_dot = [82,267]
#wavelength = np.linspace(400, 1000, 826)
wavelength = np.array(img.metadata['wavelength']).astype(float)
wavelength_index = {value: idx for idx, value in enumerate(wavelength)}
def find_wavelength(value):
    closest_value = min(wavelength, key=lambda x_val: abs(x_val - value))
    return wavelength_index[closest_value]

#for i in range(img.shape[-1]):
#    intensity1.append(img[blue_dot[0],blue_dot[1],i])
#    intensity2.append(img[orange_dot[0],orange_dot[1],i])

    
fig, (ax, ax1, ax2) = plt.subplots(ncols=3,figsize=(21,5))

ax.plot(wavelength, img_normalized[blue_dot[0], blue_dot[1], :], color='darkblue')
ax.set_xlabel("Wavelength", fontsize=20)
#ax.plot(wavelength, intensity2)
ax.plot(wavelength, img_normalized[orange_dot[0], orange_dot[1], :], color='orange')
#ax.set_xlabel("Wavelength", fontsize=20)
ax.set_title("Spectrogram Corrected", fontsize=20)
ax.set_xlim(450,900)

ax1.imshow(rbg, aspect="auto")
ax1.plot(blue_dot[1],blue_dot[0], marker="o", markersize=10, markeredgecolor="darkblue", markerfacecolor="dodgerblue")
ax1.plot(orange_dot[1],orange_dot[0], marker="o", markersize=10, markeredgecolor="darkorange", markerfacecolor="orange")
ax1.set_title("RGB JPEG in dataset", fontsize=20)   

#img_normalized = img.load()
#img_RGB = np.stack((img_normalized[:,:,int(img.metadata['default bands'][2])],img_normalized[:,:,int(img.metadata['default bands'][1])],img_normalized[:,:,int(img.metadata['default bands'][0])]), axis=2)
img_RGB = np.stack((img_normalized[:,:,int(img.metadata['default bands'][2])],img_normalized[:,:,int(img.metadata['default bands'][1])],img_normalized[:,:,int(img.metadata['default bands'][0])]), axis=2)


#print(img_RGB.shape)
ax2.imshow(img_RGB)
ax2.plot(blue_dot[1],blue_dot[0], marker="o", markersize=10, markeredgecolor="darkblue", markerfacecolor="dodgerblue")
ax2.plot(orange_dot[1],orange_dot[0], marker="o", markersize=10, markeredgecolor="darkorange", markerfacecolor="orange")
ax2.set_title("Computed Synthetic RGB", fontsize=20)


# In[ ]:





# In[152]:


### Let's find mean spectra
from tqdm import tqdm

# coef = np.zeros((img.shape[0], img.shape[1]))
# for x in tqdm(range(0, img.shape[0])):
#     for y in range(0, img.shape[1]):
#         coef[x,y] = np.sqrt(np.mean(img[x,y,:]**2))
#wavelength = wavelength[find_wavelength(450):find_wavelength(900)]
#img_normalized = img_normalized[:,:,find_wavelength(450):find_wavelength(900)]

(x,y,z) = np.where(gt==1)
# mean_normal = 0
# for i in range(0,len(x)):
#     assert(z[i]==0)
#     mean_normal += img_normalized[x[i],y[i],:]
# mean_normal /= len(x)
# #mean_normal[mean_normal<0]=0
print(len(x))
mean_normal = np.mean(img_normalized[x,y,:],axis=0)
#mean_normal = (mean_normal - np.min(mean_normal)) / (np.max((mean_normal)) - np.min(mean_normal))
#mean_normal = (mean_normal) / (np.max((mean_normal)))
#median_normal = np.median(img_normalized[x,y,:],axis=0)
std_normal = np.std(img_normalized[x,y,:],axis=0)
#lower_normal = np.percentile(img_normalized[x,y,:], 2.5, axis=0)
#upper_normal = np.percentile(img_normalized[x,y,:], 97.5, axis=0)

(x,y,z) = np.where(gt==2)
# mean_tumor = 0
# for i in range(0,len(x)):
#     assert(z[i]==0)
#     mean_tumor += img_normalized[x[i],y[i],:]
# mean_tumor /= len(x)
# #mean_tumor[mean_tumor<0]=0
print(len(x))
mean_tumor = np.mean(img_normalized[x,y,:],axis=0)
#mean_tumor = (mean_tumor - np.min(mean_tumor)) / (np.max((mean_tumor)) - np.min(mean_tumor))
#mean_tumor = (mean_tumor) / (np.max((mean_tumor)))
#median_tumor = np.median(img_normalized[x,y,:],axis=0)
std_tumor = np.std(img_normalized[x,y,:],axis=0)
#lower_tumor = np.percentile(img_normalized[x,y,:], 2.5, axis=0)
#upper_tumor = np.percentile(img_normalized[x,y,:], 97.5, axis=0)

(x,y,z) = np.where(gt==3)
print(x,y)
# mean_blood = 0
# for i in range(0,len(x)):
#     assert(z[i]==0)
#     mean_blood += img_normalized[x[i],y[i],:]
# mean_blood /= len(x)
#mean_blood[mean_blood<0]=0
mean_blood = np.mean(img_normalized[x,y,:],axis=0)
#mean_blood = (mean_blood - np.min(mean_blood)) / (np.max(mean_blood) - np.min(mean_tumor))
#mean_blood = (mean_blood) / (np.max(mean_blood))
std_blood = np.std(img_normalized[x,y,:],axis=0)
#median_blood = np.median(img_normalized[x,y,:],axis=0)
#lower_blood = np.percentile(img_normalized[x,y,:], 2.5, axis=0)
#upper_blood = np.percentile(img_normalized[x,y,:], 97.5, axis=0)
print(len(x))

img_normalized[x,y,:].shape


# In[153]:


plt.figure()
plt.plot(wavelength, mean_normal, color = 'green', label='Normal (Mean)')
#plt.plot(wavelength, median_normal, color = 'green', label='Normal (Mean)')
plt.plot(wavelength, mean_normal + std_normal, '--', color = 'green')
plt.plot(wavelength, mean_normal - std_normal, '--', color = 'green')
plt.fill_between(wavelength, mean_normal - std_normal, mean_normal + std_normal, color='green', alpha=0.2, label='Normal (Std)')
plt.plot(wavelength, mean_tumor, color = 'red', label='Tumor (Mean)')
#plt.plot(wavelength, median_tumor, color = 'red', label='Tumor (Mean)')
plt.plot(wavelength, mean_tumor + std_tumor, '--', color = 'red')
plt.plot(wavelength, mean_tumor - std_tumor, '--', color = 'red')
plt.fill_between(wavelength, mean_tumor - std_tumor, mean_tumor + std_tumor, color='red', alpha=0.2, label='Tumor Std')
plt.plot(wavelength, mean_blood, color = 'blue', label='Blood (Mean)')
#plt.plot(wavelength, median_blood, color = 'blue', label='Blood (Mean)')
plt.plot(wavelength, mean_blood - std_blood, '--', color = 'blue')
plt.plot(wavelength, mean_blood + std_blood, '--', color = 'blue')
plt.fill_between(wavelength, mean_blood - std_blood, mean_blood + std_blood, color='blue', alpha=0.2, label='Blood Std')
plt.xlim(450,900)
plt.legend()


# In[183]:


### Visualising spectrograms with calibration
#wavelength = np.linspace(400, 1000, 826)

#fig, ax2 = plt.subplots(ncols=1,figsize=(5,5))

#molecules, x = read_molecules(520, 900, wavelength)
#molecules, x = read_molecules(580, 800, wavelength)

molecules, x = read_molecules(599, 800, wavelength)
#molecules, x = read_molecules(520, 900, wavelength)

fig, (ax, ax1, ax2) = plt.subplots(ncols=3,figsize=(15,5))

relative_attenuation = (-np.log((img_normalized[orange_dot[0], orange_dot[1], :]) / (img_normalized[blue_dot[0], blue_dot[1], :])))[np.in1d(wavelength,x)]
#relative_attenuation = (-np.log((img_normalized[orange_dot[0], orange_dot[1], :]) / (img_normalized[blue_dot[0], blue_dot[1], :])))
#relative_attenuation = ((-np.log(((img_normalized[blue_dot[0], blue_dot[1], :])) / (img_normalized[orange_dot[0], orange_dot[1], :]))))



ax.plot(wavelength, (img_normalized[blue_dot[0], blue_dot[1], :]), color='darkblue')
#ax.plot(wavelength, intensity2)
ax1.plot(wavelength, (img_normalized[orange_dot[0], orange_dot[1], :]), color='orange')

#ax2.plot(wavelength, relative_attenuation)
ax2.plot(x, relative_attenuation)
ax2.set_xlabel("Wavelength", fontsize=20)
#ax.set_xlabel("Wavelength", fontsize=20)
ax2.set_title("Relative Attenuation", fontsize=20)
#ax2.set_xlim(520,950)


# In[156]:


### Visualising dark and white reference

# white_path = path+"/{}/whiteReference.hdr".format(patient)
# white = open_image(white_path)
# white = white.load()

# dark_path = path+"/{}/darkReference.hdr".format(patient)
# dark = open_image(dark_path)
# dark = dark.load()

# print(dark.shape)

# fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(10,5))

# ax.plot(white[:,:,0].squeeze(0))
# ax.set_title("White reference", fontsize=20)

# ax1.plot(white[:,:,825].squeeze(0))
# ax1.set_title("Dark reference", fontsize=20)


# In[157]:


### Visualising spectrograms with calibration

# intensity1_c = []
# intensity2_c = []

# wavelength = np.linspace(400, 1000, 826)
# #calb = np.empty((img.shape[0], img.shape[1], img.shape[2]))


# for i in range(img.shape[-1]):
#     # Raw data for white and dark calibration references comes as (1,330,826) array. It means we have 1d vector
#     # for each of the 826 wavelengths.    
#     # Here we form a 2d array (to match the HSI image size) for white and dark references out of the 1d data 
#     # by repeating the 1d vector 389 times for the current wavelength in the loop. 
#     # (1,330,826) --> (389,330) 

#     white_full = np.tile(white[:,:,i].squeeze(0).squeeze(1), (img.shape[0],1))
#     dark_full  = np.tile(dark[:,:,i].squeeze(0).squeeze(1), (img.shape[0],1))
    
#     # We wanna calibrate our HSI images w.r.t. white reference. Why? Because we saw by looking at the raw data that different 
#     # wavelengths have different reflection from the white materila (i.e. reflecting 99% of light). So
#     # we calibrate our HSI images 
    
#     intensity1_c.append((img[blue_dot[0],blue_dot[1],i] - dark_full[blue_dot[0],blue_dot[1]])/(white_full[blue_dot[0],blue_dot[1]]-dark_full[blue_dot[0],blue_dot[1]]))
#     intensity2_c.append((img[orange_dot[0],orange_dot[1],i] - dark_full[orange_dot[0],orange_dot[1]])/(white_full[orange_dot[0],orange_dot[1]]-dark_full[orange_dot[0],orange_dot[1]]))
#     #calb[:,:,i] = (img[:,:,i].squeeze(2) - dark_full)/(white_full-dark_full) 
#     #calb = (img[:,:,i].squeeze(2) - dark_full)/(white_full-dark_full) 
# #print(calb.shape)
    
# fig, (ax, ax1, ax2) = plt.subplots(ncols=3,figsize=(15,5))

# ax.plot(wavelength, intensity1)
# ax.set_xlabel("Wavelength", fontsize=20)
# ax.plot(wavelength, intensity2)
# #ax.set_xlabel("Wavelength", fontsize=20)
# ax.set_title("Spectrogram", fontsize=20)
# ax.set_xlim(400,900)

# ax1.plot(wavelength, intensity1_c)
# ax1.set_xlabel("Wavelength", fontsize=20)
# ax1.plot(wavelength, intensity2_c)
# #ax.set_xlabel("Wavelength", fontsize=20)
# ax1.set_title("Spectrogram corrected", fontsize=20)
# ax1.set_xlim(400,900)

# molecules, x = read_molecules(520, 900, wavelength)
# relative_attenuation = (-np.log(np.array(intensity2_c) / np.array(intensity1_c)))[np.in1d(wavelength,x)]
# ax2.plot(x, relative_attenuation)
# ax2.set_xlabel("Wavelength", fontsize=20)
# #ax.set_xlabel("Wavelength", fontsize=20)
# ax2.set_title("Relative Attenuation", fontsize=20)
# ax2.set_xlim(520,900)


# In[158]:


#wavelength


# # Now we will build the compositions (absorption and scattering)

# ## First, we do absorption

# In[159]:


# Calculate the absorption coefficient [cm^-1] of the tissue for the given wavelength:
# mu_a=mu_a_HbO2+mu_a_HHb+mu_a_H2O+mu_a_fat+mu_a_oxCCO+mu_a_redCCO


# In[160]:


# path_absorp = "/home/kevinscibilia/UCL-NIR-Spectra/spectra/"
# hbo2_absorp = path_absorp + "hb02.txt"
# hhb_absorp = path_absorp + "hb.txt"
# water_absorp = path_absorp + "matcher94_nir_water_37.txt"
# diff_cyto_absorp = path_absorp + "cytoxidase_diff_odmMcm.txt"
# fat_absorp = path_absorp + "fat.txt"
# cyto_oxy_absorp = path_absorp + "moody cyt aa3 oxidised.txt"
# cyto_red_absorp = path_absorp + "moody cyt aa3 reduced.txt"


# In[ ]:





# In[161]:


### reading cpectra from .txt
# def read_spectra(file_name):
#     with open(file_name, 'r') as data:
#         x = []
#         y = []
#         for line in data:
#             p = line.split()
#             x.append(float(p[0]))
#             y.append(float(p[1]))

#     return x, y

# x_cox, y_cox = read_spectra(cyto_oxy_absorp) # cytochrome oxydised
# x_cred, y_cred = read_spectra(cyto_red_absorp) # cytochrome reduced
# x_hbo2, y_hbo2 = read_spectra(hbo2_absorp) # oxyhemoglobin
# x_hbb, y_hbb = read_spectra(hhb_absorp) # hemoglobin
# #x_water, y_water = read_spectra(water_absorp) # water
# #x_fat, y_fat = read_spectra(fat_absorp)

# y_cox = np.array([i * 2.3025851 for i in y_cox]) # from extinction to absorption  
# y_cred = np.array([i * 2.3025851 for i in y_cred]) # from extinction to absorption  
# #y_water = [i * 2.3025851 for i in y_water] # from extinction to absorption  
# y_hbo2 = np.array([i * 10 * 1000 for i in y_hbo2]) # from mm and micromole to cm and minimole 
# y_hbb = np.array([i * 10 * 1000 for i in y_hbb]) # from mm and micromole to cm and minimole
# #y_fat = [i / 100 for i in y_fat] # from m to cm 


# # cutting all spectra to the range 650 - 1000 nm
# left_cut = 520
# right_cut = 999

# import bisect
# # ix_left = x_cox.index(left_cut)
# # ix_right = x_cox.index(right_cut)
# # x = x_cox[bisect.bisect_left(x_cox, left_cut):bisect.bisect_right(x_cox, right_cut)]
# # y_cox = y_cox[ix_left:ix_right+1]

# # ix_left = x_cred.index(left_cut)
# # ix_right = x_cred.index(right_cut)
# # y_cred = y_cred[ix_left:ix_right+1]

# # ix_left = x_water.index(left_cut)
# # ix_right = x_water.index(right_cut)
# # y_water = y_water[ix_left:ix_right+1]

# # ix_left = x_hbo2.index(left_cut)
# # ix_right = x_hbo2.index(right_cut)
# # y_hbo2 = y_hbo2[ix_left:ix_right+1]

# # ix_left = x_hbb.index(left_cut)
# # ix_right = x_hbb.index(right_cut)
# # y_hbb = y_hbb[ix_left:ix_right+1]

# # ix_left = x_fat.index(left_cut)
# # ix_right = x_fat.index(right_cut)
# # y_fat = y_fat[ix_left:ix_right+1]


# ### Visualising spectrograms of chromophores
# fig, (ax, ax1) = plt.subplots(ncols=2,figsize=(10,5))

# ax.plot(x_cox, y_cox, label='cy_ox')
# ax.set_xlabel("Wavelength", fontsize=20)
# ax.plot(x_cred, y_cred, label='cy_red')
# assert(np.all(x_cred == x_cox))
# ax.plot(x_cox, y_cox - y_cred, label='cy_diff')
# ax.plot(x_hbo2, y_hbo2, label='hbo2')
# ax.plot(x_hbb, y_hbb, label='hbb')
# #ax.plot(x_water, y_water, label='water')
# #ax.plot(x_fat, y_fat, label='fat')
# ax.set_title("Spectrogram", fontsize=20)
# ax.set_xlim(600, 950)
# #ax.set_ylim(0, 55)
# #ax.set_ylim(0, 15)
# ax.legend(fontsize=14)


# # ax1.plot(x_water, y_water, label='water')
# # ax1.set_xlabel("Wavelength", fontsize=20)
# # ax1.plot(x_fat, y_fat, label='fat')
# # #ax.set_xlabel("Wavelength", fontsize=20)
# # ax1.set_title("Spectrogram", fontsize=20)
# # ax1.set_xlim(500, 950)
# # ax1.set_ylim(0, 0.6)
# # ax1.legend(fontsize=20)


# #maxpicks = {"w": np.max(y_water), "f": np.max(y_fat), "hbo2": np.max(y_hbo2), "hbb": np.max(y_hbb), 
# #            "cox": np.max(y_cox), "cred":np.max(y_cred)}

# maxpicks = {"hbo2": np.max(y_hbo2), "hbb": np.max(y_hbb), 
#             "cox": np.max(y_cox), "cred":np.max(y_cred)}


# #maxpicks


# In[184]:


#filtered_wavelength = [w for w in wavelength if w >= 520 and w <= 899]
fig, (ax, ax1) = plt.subplots(ncols=2,figsize=(10,5))

y_hbo2_f, y_hb_f, y_coxa, y_creda, y_c_oxy, y_c_red, y_b_oxy, y_b_red, y_fat, y_water = molecules #, y_water, y_fat = molecules
#y_hbo2_f, y_hb_f = molecules

#y_hbo2_f, y_hb_f, y_coxa, y_creda, y_water = molecules 
ax.plot(x, y_hbo2_f, label='HbO_2')
ax.plot(x, y_hb_f, label='Hbb')
ax.plot(x, y_coxa, label='oxCCO')
ax.plot(x, y_creda, label='redCCO')
#ax.plot(x, y_water, label='water')
#ax1.plot(x, y_water, label='water')
#ax1.plot(x, y_fat, label='fat')

ax1.plot(x, y_c_oxy - y_c_red, label='diffCC')
ax1.plot(x, y_b_oxy - y_b_red, label='diffCB')
ax1.plot(x, y_coxa - y_creda, label='diffCCO')
ax1.plot(x, y_fat, label='Fat')
# ax1.plot(x, y_water, label='Water')

ax.legend()
ax1.legend()
plt.title("Spectra")


# In[191]:


from scipy.linalg import pinv
# M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
#                             np.asarray(y_hb_f),
#                             np.asarray(y_coxa - y_creda),
#                             np.asarray(y_water),
#                             np.asarray(y_fat))))
b_blood = 0.66
#b_blood = 0.25
#a_blood = 22 #[1/cm]
a_blood=1

b_brain = 1.611
#b_brain=1.2
#b_brain = 1.629
#b_brain = 1.629
#a_brain = 24.2
a_brain = 1

#b_tumor = 3.254
b_tumor = 0.334
a_tumor = 1

b_gm = 0.334
a_gm = 1

M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                            np.asarray(y_hb_f),
                            np.asarray(y_coxa - y_creda),
                            np.asarray(y_water),
                            np.asarray(y_fat))))
#                            np.asarray(y_fat))))

print(M.shape)
mu_s_r_blood = (a_blood * np.power(np.divide(x,500),-b_blood)) / (1 - 0.9)
mu_s_r_brain = (a_brain * np.power(np.divide(x,500),-b_brain)) / (1 - 0.9)
mu_s_r_tumor = (a_tumor * np.power(np.divide(x,500),-b_tumor)) / (1 - 0.9)
mu_s_r_gm = (a_gm * np.power(np.divide(x,500),-b_gm)) / (1 - 0.9)

print(mu_s_r_blood.shape)
M = np.hstack((M, mu_s_r_blood[:, np.newaxis]))
#M = np.hstack((M, mu_s_r_brain[:, np.newaxis]))

#M = np.hstack((M, mu_s_r_tumor[:, np.newaxis]))
#M = np.hstack((M, mu_s_r_gm[:, np.newaxis]))
print(M.shape)

M_pinv = pinv(M)
concentrations = (M_pinv @ relative_attenuation)
plt.plot(x, M @ concentrations)
plt.plot(x, relative_attenuation)
print(concentrations)


# In[198]:


import random
from tqdm.auto import tqdm

#blood_index = 1300
#orange_dot = [x_blood[blood_index], y_blood[blood_index]]
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

reference_index_normal = random.randint(0, len(x_normal) - 1)
reference_point = [x_normal[reference_index_normal], y_normal[reference_index_normal]]
#reference = (img_normalized[reference_point[0], y_normal[reference_point[1]], :])
reference = (img_normalized[reference_point[0], reference_point[1], :])
#reference_index_blood = random.randint(0, len(x_blood) - 1)
#reference = img_normalized[x_blood[reference_index_blood], y_blood[reference_index_blood], :]
# reference_index_tumor = random.randint(0, len(x_tumor) - 1)
# reference = img_normalized[x_tumor[reference_index_tumor], y_tumor[reference_index_tumor], :]



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
        # hbt_inferred[i,j] = concentrations[0] + concentrations[1]
        # hbdiff_inferred[i,j] = concentrations[0] - concentrations[1]
        # diffCCO_inferred[i,j] = concentrations[2] - concentrations[3]
        # CCO_inferred[i,j] = concentrations[2] + concentrations[3]
        # blood_scattering_inferred[i,j] = concentrations[4]
        # brain_scattering_inferred[i,j] = concentrations[5]
        
        hbt_inferred[i,j] = concentrations[0] + concentrations[1]
        hbdiff_inferred[i,j] = concentrations[0] - concentrations[1]
        diffCCO_inferred[i,j] = concentrations[2]
        blood_scattering_inferred[i,j] = concentrations[5]
        brain_scattering_inferred[i,j] = concentrations[4]

        # hbt_inferred[i,j] = concentrations[0] + concentrations[1]
        # hbdiff_inferred[i,j] = concentrations[0] - concentrations[1]
        # diffCCO_inferred[i,j] = concentrations[2]
        # blood_scattering_inferred[i,j] = concentrations[4]
        # brain_scattering_inferred[i,j] = concentrations[5]


# In[199]:


from matplotlib.colors import Normalize
error_tolerance = 0.1
(x_normal,y_normal) = np.where((np.squeeze(gt)==1) & (error_inferred < error_tolerance))
(x_tumor,y_tumor) = np.where((np.squeeze(gt)==2) & (error_inferred < error_tolerance))
(x_blood,y_blood) =  np.where((np.squeeze(gt)==3) & (error_inferred < error_tolerance))

fig, axs = plt.subplots(nrows=4, ncols=3,figsize=(15,15))
ax_RGB, ax_hbt, ax_cco, ax_bloodscatter, ax_brainscatter, ax_error, ax_greenpoint, ax_bluepoint, ax_yellowpoint, ax_bottom1, ax_bottom2, ax_bottom3 = axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2], axs[2,0], axs[2,1], axs[2,2], axs[3,0], axs[3,1], axs[3,2]
ax_RGB.imshow(img_RGB)
ax_RGB.set_title("Synthetic RGB", fontsize=20)
ax_RGB.imshow(mask1, cmap=cm.tab10)
ax_RGB.imshow(mask2, cmap=cm.Dark2)
ax_RGB.imshow(mask3, cmap=cm.get_cmap('spring_r'))

hbt_filtered = hbt_inferred
hbt_filtered = np.where(error_inferred > error_tolerance, np.min(hbt_inferred), hbt_filtered)
#hbt_filtered = np.where(hbt_filtered < 0.0, 0, hbt_filtered)
hbt_img = ax_hbt.imshow(hbt_filtered, cmap='Reds')
fig.colorbar(hbt_img, ax=ax_hbt)
#ax_hbt.set_title("Inferred HbT > 0 \n Fitting MAE < " + str(error_tolerance), fontsize=15)
ax_hbt.set_title("Inferred HbT \n Fitting MAE < " + str(error_tolerance), fontsize=15)

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
ax_cco.set_title("Inferred HbDiff > 0  \n Fitting MAE < " + str(error_tolerance), fontsize=15)

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
ax_bloodscatter.set_title("Inferred blood scattering", fontsize=13)
fig.colorbar(blood_scattering_img, ax=ax_bloodscatter)

brain_scattering_inferred_filtered = brain_scattering_inferred
#brain_scattering_inferred_filtered = np.where(brain_scattering_inferred_filtered < 0.0, 0, brain_scattering_inferred_filtered)
#brain_scattering_inferred_filtered = np.where(error_inferred > error_tolerance, np.nan, brain_scattering_inferred_filtered)
brain_scattering_img = ax_brainscatter.imshow((brain_scattering_inferred_filtered), cmap='seismic_r')
#ax_brainscatter.set_title("Inferred brain scattering", fontsize=13)
ax_brainscatter.set_title("Inferred brain scattering", fontsize=13)
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
diffCCO_inferred_filtered = np.where(diffCCO_inferred_filtered < 0, 0, diffCCO_inferred_filtered)
#diffCCO_img = ax_error.imshow(diffCCO_inferred_filtered, norm=Normalize(vmin=0, vmax=diffCCO_inferred_filtered.max()/4))
diffCCO_img = ax_error.imshow(diffCCO_inferred_filtered)
fig.colorbar(diffCCO_img, ax=ax_error)
ax_error.set_title("diffCCO > 0(MAE < " + str(error_tolerance) +")")

rows, cols = np.indices(error_inferred.shape)
#points_1 = np.where((error_inferred <= error_tolerance*0.25) & (hbt_inferred > 0.005) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
#points_1 = np.where((error_inferred <= error_tolerance) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
green_index = random.randint(0, len(x_normal) - 1)
#green_index = reference_index_normal
green_point = [x_normal[green_index], y_normal[green_index]]

yellow_index = random.randint(0, len(x_blood) - 1)
yellow_point = [x_blood[yellow_index], y_blood[yellow_index]]

ax_RGB.plot(green_point[1], green_point[0], marker="o", markersize=5, markeredgecolor="darkgreen", markerfacecolor='#66FF66')
#points_1[0].shape

#points_2 = np.where((hbt_inferred > 0.005) & (error_inferred >= error_tolerance*0.9) & (error_inferred <= error_tolerance) & (hbt_inferred > 0.005) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
#points_2 = np.where((error_inferred >= error_tolerance*0.75) & (error_inferred <= error_tolerance) & (rows > 150) & (rows < 250) & (cols > 150) & (cols < 250))
#orange_point = [points_2[0][0], points_2[1][0]]

#(x_normal,y_normal) = np.where((np.squeeze(gt)==1) & (error_inferred < error_tolerance))
#(x_tumor,y_tumor) = np.where((np.squeeze(gt)==2) & (error_inferred < error_tolerance))

(x_normal,y_normal) = np.where((np.squeeze(gt)==1) & (error_inferred < error_tolerance))
(x_tumor,y_tumor) = np.where((np.squeeze(gt)==2) & (error_inferred < error_tolerance))
(x_blood,y_blood) =  np.where((np.squeeze(gt)==3) & (error_inferred < error_tolerance))

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

ax_RGB.plot(reference_point[1], reference_point[0], marker="x", markersize=8, markeredgecolor="orange", markerfacecolor="#FFA500")


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
    ax_RGB.plot(blue_point[1], blue_point[0], marker="o", markersize=5, markeredgecolor="darkblue", markerfacecolor="#0000FF")
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




# In[200]:


img_normalized.shape
img_normalized.reshape(-1, img_normalized.shape[-1]).reshape(img_normalized.shape)[:,:,np.in1d(wavelength,x)].shape
relative_attenuation[np.newaxis, :].shape


# In[201]:


config.molecule_count = 3
config.max_b = 4
config.max_a = 100

def S(a_ti, b_ti, b_t1, a_t1, x):
    result = ((((x/500)**(-b_ti)) * a_ti) - (((x/500)**(-b_t1)) * a_t1)) / (1-0.9)
    #result = ((((x/500)**(-b_t1)) * a_ti) - (((x/500)**(-b_t1)) * a_t1)) / (1-0.9)
    #result = ((((np.power(np.divide(x,500),-b_t1))) * a_ti) - (((np.power(np.divide(x,500),-b_t1))) * a_t1)) / (1-0.9)

    return result

def f(X,*arg):
    m = config.molecule_count
    b = arg[0]
    b_t1, a_t1 = arg[1], arg[2]
    M = arg[3]
    x = arg[4]
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
    left_bound = np.append(np.ones(config.molecule_count)*(-np.inf), [-np.inf, 0])
    right_bound = np.append(np.ones(config.molecule_count)*np.inf, [np.inf, config.max_b])
    
    current_x = np.zeros(config.molecule_count+2)
    #current_x[-2] = 0
    current_x[-2] = a_t1
    current_x[-1] = b_t1
    
    #coef_list = []
    #scattering_params_list = []
    #errors_scatter = []
    coef_list = np.zeros((b.shape[0], config.molecule_count))
    scattering_params_list = np.zeros((b.shape[0], 2))
    errors_scatter = np.zeros((b.shape[0]))
    
    
    for i in tqdm(range(0, b.shape[0])):
        result = scipy.optimize.least_squares(f, current_x, args=(b[i], b_t1, a_t1, M, x), bounds=(left_bound, right_bound))
        #current_x = result.x
        coef_list[i,:] = (result.x[:config.molecule_count])
        scattering_params_list[i,:] = result.x[config.molecule_count:]
        errors_scatter[i] = np.sqrt(2*result.cost)
        
    error = sum(errors_scatter) / len(errors_scatter)
    return error, coef_list, scattering_params_list, errors_scatter
    
def helicoid_optimisation_ti_error(params_t1, *args):
    error, _, _, _ = helicoid_optimisation_ti(params_t1, *args)
    return error

def helicoid_optimisation_scattering(img_normalized, reference_point, M, x):
    img_ref = -np.log(img_normalized[::7,::7,:] / img_normalized[reference_point[0], reference_point[1], :])[:,:,np.in1d(wavelength,x)]
    b = img_ref.reshape(-1, img_ref.shape[-1]) #reshape to 1D x wavelength
    #b = relative_attenuation[np.newaxis, :]
    #print(b.shape)
    
    directory = './helicoid/scattering/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    M_pinv = pinv(M)
    # print(spectr2.shape)
    #b = spectr2 / spectr1
    #b = np.log(1 / np.asarray(b))
    
    result = scipy.optimize.brute(helicoid_optimisation_ti_error, (slice(0,config.max_a, config.max_a/5), slice(0,config.max_b, config.max_b/150)), args=(b,M,x), finish=None, full_output=True, workers=4)    
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
    #plt.savefig(directory+'/'+'error')
    
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
    return coef_list, scattering_params, errors, a_t1, b_t1


# In[202]:


hbt_opti = np.zeros((img.shape[0], img.shape[1]))
hbdiff_opti = np.zeros((img.shape[0], img.shape[1]))
error_opti = np.zeros((img.shape[0], img.shape[1]))
diffCCO_opti = np.zeros((img.shape[0], img.shape[1]))


M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                            np.asarray(y_hb_f),
                            np.asarray(y_coxa - y_creda))))

coefs, scattering_params, errors, a_t1, b_t1 = helicoid_optimisation_scattering(img_normalized, reference_point, M, x)


# In[203]:


#plt.imshow(img_RGB[::7,::7])
#plt.imshow(errors[:,:])
diffCCO_scatter_filtered = scattering_params[:,:,0]
#diffCCO_scatter_filtered = coefs[:,:,2]
#diffCCO_scatter_filtered = errors
#diffCCO_scatter_filtered = np.where(errors > 2, np.min(diffCCO_scatter_filtered), diffCCO_scatter_filtered)
#diffCCO_scatter_filtered = np.where(diffCCO_scatter_filtered < 0.0, 0, diffCCO_scatter_filtered)

plt.imshow(diffCCO_scatter_filtered)
plt.colorbar()
#img_RGB[::3,::3].shape


# In[207]:


b_t1


# In[204]:


img_ref = -np.log((img_normalized[::7,::7,:] / img_normalized[reference_point[0], reference_point[1], :])[:,:,np.in1d(wavelength,x)])
# scattering = ((((np.power(np.divide(x,500),-0.66))))) / (1-0.9)
# #print(M.shape)
# #print(scattering.shape)
# M_ext = np.hstack((M, scattering[:, np.newaxis]))

# point = [15,33]

# plt.plot(x, M_ext @ np.concatenate((coefs[point[0], point[1], :].flatten(), scattering_params[point[0],point[1],0].flatten()), axis=-1))
# plt.plot(x, img_ref[point[0], point[1], :].flatten())


# In[209]:


hbt_inferred = coefs[:,:,0] + coefs[:,:,1]
hbdiff_inferred = coefs[:,:,0] - coefs[:,:,1]
diffCCO_inferred = coefs[:,:,2]
blood_scattering_inferred = scattering_params[:,:,0]
brain_scattering_inferred = scattering_params[:,:,1]
error_inferred = errors

img_RGB = np.stack((img_normalized[::7,::7,int(img.metadata['default bands'][2])],img_normalized[::7,::7,int(img.metadata['default bands'][1])],img_normalized[::7,::7,int(img.metadata['default bands'][0])]), axis=2)

error_tolerance = 1.5
(x_normal,y_normal) = np.where((np.squeeze(gt[::7,::7,:])==1) & (error_inferred < error_tolerance))
(x_tumor,y_tumor) = np.where((np.squeeze(gt[::7,::7,:])==2) & (error_inferred < error_tolerance))
(x_blood,y_blood) =  np.where((np.squeeze(gt[::7,::7,:])==3) & (error_inferred < error_tolerance))

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
diffCCO_inferred_filtered = np.where(diffCCO_inferred_filtered < 0, 0, diffCCO_inferred_filtered)
#diffCCO_img = ax_error.imshow(diffCCO_inferred_filtered, norm=Normalize(vmin=0, vmax=diffCCO_inferred_filtered.max()/4))
diffCCO_img = ax_error.imshow(diffCCO_inferred_filtered)
fig.colorbar(diffCCO_img, ax=ax_error)
ax_error.set_title("diffCCO > 0(E < " + str(error_tolerance) +")")

rows, cols = np.indices(error_inferred.shape)
#points_1 = np.where((error_inferred <= error_tolerance*0.25) & (hbt_inferred > 0.005) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
#points_1 = np.where((error_inferred <= error_tolerance) & (rows > 100) & (rows < 250) & (cols > 100) & (cols < 250))
green_index = random.randint(0, len(x_normal) - 1)
#green_index = reference_index_normal
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

(x_normal,y_normal) = np.where((np.squeeze(gt[::7,::7,:])==1) & (error_inferred < error_tolerance))
(x_tumor,y_tumor) = np.where((np.squeeze(gt[::7,::7,:])==2) & (error_inferred < error_tolerance))
(x_blood,y_blood) =  np.where((np.squeeze(gt[::7,::7,:])==3) & (error_inferred < error_tolerance))

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



# In[ ]:





# In[29]:


# ### reading spectra from Luca's paper directly
# ### careful about water absorption (I guess it is extinction instead)

# def read_spectra_all(file_name):
#     with open(file_name, 'r') as data:
#         x = []
#         y1 = []
#         y2 = []
#         y3 = []
#         y4 = []
#         y5 = []
#         y6 = []
#         y7 = []
#         for line in data:
#             p = line.split()
#             x.append(float(p[0]))
#             y1.append(float(p[1]))
#             y2.append(float(p[2]))
#             y3.append(float(p[3]))
#             y4.append(float(p[4]))
#             y5.append(float(p[5]))
#             y6.append(float(p[6]))
#             y7.append(float(p[7]))

#     return x, y1, y2, y3, y4, y5, y6, y7

# x1, yw, yf, yhox, yh, ycox, ycred, ydiff = read_spectra_all("/home/ivan/aimlab/UCL-NIR-Spectra/spectra/all-in-one.txt")

# yhox = [i * 2.3025851 / 1000 for i in yhox] # from extinction to absorption  
# yh = [i * 2.3025851 / 1000 for i in yh] # from extinction to absorption  
# ycox = [i * 2.3025851 / 1000 for i in ycox] # from extinction to absorption  
# ycred = [i * 2.3025851 / 1000 for i in ycred] # from mm and micromole to cm and minimole 

# ### Visualising spectrograms of chromophores from Luca's paper
# fig, (ax, ax1) = plt.subplots(ncols=2,figsize=(10,5))

# ax.plot(x1, yw)
# ax.set_xlabel("Wavelength", fontsize=20)
# ax.plot(x1, yf)
# ax.plot(x1, yhox)
# ax.plot(x1, yh)
# ax.plot(x1, ycox)
# ax.plot(x1, ycred)
# ax.set_title("Spectrogram", fontsize=20)
# #ax.set_xlim(left_cut,right_cut)
# ax.set_ylim(0, 16)


# ax1.set_xlabel("Wavelength", fontsize=20)

# ax1.plot(x_water, y_water, label='water')
# ax1.plot(x_fat, y_fat, label='fat')
# ax1.plot(x_hbo2, y_hbo2, label='hbo2')
# ax1.plot(x_hbb, y_hbb, label='hbb')
# ax1.plot(x_cox, y_cox, label='cox')
# ax1.plot(x_cred, y_cred, label='cred')

# ax1.set_title("Spectrogram", fontsize=20)
# #ax1.set_xlim(770, 910)
# ax1.set_ylim(0, 16)
# ax1.legend()


# ## Now scattering

# In[ ]:





# # Questions:
# 
# 1. MCMM and parametric optimisation (both local or global) are heavy coputationally.
# 
# Remedy: can do MLP, but need one-to-one mapping
# 
# 2. The ill-posedness of the composition inference problem.
# 
# Remedy:
# - can we reduce to say 3 number of components (weights of the components are important though)?
# - what are physiologically plausible bounds?
# - what are the ranges of wavelength we can phisically cover ( e.g. water has a distinct peak close to 1000nm)
# - can we inject some contrast agents to get sharp spectra peaks from them (or its against the ain HyperProbe idea)?
# 
# 3. Dictionary between composition coefficients and biomarkers?
# 4. Where to get g (anisotrophy) for white,grey matter and blood?
# 5. Why cytochrome spectra are different (e.g. Moody is quite different from Cooper), which one to use?
#  
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




