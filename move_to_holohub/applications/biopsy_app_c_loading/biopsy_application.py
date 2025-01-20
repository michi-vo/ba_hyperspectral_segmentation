import sys
import os

from optimisation_helicoid import read_molecules_creatis, read_molecules_cytochrome_cb, helicoid_optimisation_ti_parallel
# sys.path.append('../../')

import scipy
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pysptools
import pysptools.spectro
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from scipy.linalg import pinv
# from optimisation_helicoid import read_molecules_creatis, read_molecules_cytochrome_cb
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.conditions import CountCondition, PeriodicCondition

import blosc
import json
import os

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def save_optimization_data(img_ref, reference_point, coef_list, scattering_params, errors, t1, time_taken, folder):
    """
    Used after running the optimisation script/notebook to be used later by the Neural Network
    """
    img_ref = torch.from_numpy(img_ref)
    reference_point = torch.from_numpy(reference_point)
    coef_list = torch.from_numpy(coef_list)
    scattering_params = torch.from_numpy(scattering_params)
    errors = np.asarray(errors)
    errors = torch.from_numpy(errors)
    t1 = torch.from_numpy(t1)
    time_taken = torch.from_numpy(np.array([time_taken]))
    
    path = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/hyperprobe_biopsies"
    
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

    # Visualize Wavelengths

    #filename = "/home/kevin/hyperprobe_biopsies/Biopsy_S2_reflectance.mat"

class loadDataOp(Operator):
    def __init__(self, *args, **kwargs):
        # condition = PeriodicCondition(self, 1000)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("biopsy_ref")
        spec.output("biopsy")

    def compute(self, op_input, op_output, context):
        global biopsy_name
        print("[loadDataOp]: Loading biopsy probe file")
        # Load Biopsy probe as reference, used to cancel out noise later
        biopsy_ref_name = "Biopsy_S1_reflectance"
        filename = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/hyperprobe_biopsies/" + biopsy_ref_name + ".mat"
        # with h5py.File(filename, 'r') as f:
        #     biopsy_ref = np.array(f['Ref_hyper'])
        biopsy_ref = decompress_large_array("/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/biopsy_1_compressed")
        biopsy = decompress_large_array("/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/biopsy_2_compressed")
        # Load Biopsy probe in question
        biopsy_name = "Biopsy_S2_reflectance"
        filename = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/hyperprobe_biopsies/" + biopsy_name + ".mat"
        # with h5py.File(filename, 'r') as f:
        #     biopsy = np.array(f['Ref_hyper'])

        op_output.emit(biopsy_ref, "biopsy_ref")
        op_output.emit(biopsy, "biopsy")
        print("[loadDataOp]: Sent data")

class preProcessDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.input("biopsy_ref")
        spec.input("biopsy")
        
        spec.output("img")
        spec.output("img_ref")
        spec.output("b")
        spec.output("M")
        spec.output("x")
        self.wavelengths = np.linspace(510,900,79)

    def compute(self, op_input, op_output, context):
        biopsy = op_input.receive("biopsy")
        biopsy_ref = op_input.receive("biopsy_ref")

        print("[preProcessDataOp]: Received data")
        M, x = self.create_absorption_matrix()
        print("[preProcessDataOp]: Successfully created absorption matrix")

        # Convert cube
        biopsy_ref_transposed = np.transpose(biopsy_ref)
        biopsy_ref_transposed.shape
        biopsy_ref_transposed[biopsy_ref_transposed<=0] = 10**-3
        biopsy_ref_transposed = F.avg_pool2d(torch.tensor(biopsy_ref_transposed).permute(2, 0, 1).unsqueeze(0), kernel_size=4, stride=4).squeeze(0).permute(1, 2, 0).numpy()
        mean_biopsy_s1 = np.mean(biopsy_ref_transposed[150:350, 150:350, :].reshape(-1, biopsy_ref_transposed.shape[-1]), axis=0)

        biopsy_transposed = np.transpose(biopsy)
        biopsy_transposed.shape
        biopsy_transposed[biopsy_transposed<=0] = 10**-3
        biopsy_transposed = F.avg_pool2d(torch.tensor(biopsy_transposed).permute(2, 0, 1).unsqueeze(0), kernel_size=4, stride=4).squeeze(0).permute(1, 2, 0).numpy()
        img = biopsy_transposed
        
        # Combine probes to filter out noise and enhance contrast with log
        # Reduces spatial resolution of img with stepsize=coarseness while keeping all spectral bands (img[::coarseness,::coarseness,:]) 
        coarseness=1
        img_ref = -np.log(img[::coarseness,::coarseness,:] / mean_biopsy_s1)[:,:,np.in1d(self.wavelengths,x)]
        b = img_ref.reshape(-1, img_ref.shape[-1])

        # construct rgb
        # semisynthetic_rgb = np.concatenate((normalize(data_transposed[:,:,23][:,:,np.newaxis]), normalize(data_transposed[:,:,6][:,:,np.newaxis]), normalize(data_transposed[:,:,0][:,:,np.newaxis])),axis=2)
        # print(semisynthetic_rgb.shape)
        # rgb_channels = [23, 6, 0]  # Replace these with the desired indices
        # rgb_data = np.stack([data_transposed[:, :, i] for i in rgb_channels], axis=-1)

        # Normalize data to [0, 1]
        # rgb_data_normalized = normalize(rgb_data) #necessary?
        # print("finished processing, sending to viz")
        # # Emit this data
        # op_output.emit(rgb_data_normalized.astype(np.float32), "out")

        op_output.emit(img, "img")
        op_output.emit(img_ref, "img_ref")
        op_output.emit(b, "b")
        op_output.emit(M, "M")
        op_output.emit(x, "x")

        print("[preProcessDataOp]: Sent data")

    def create_absorption_matrix(self):
        # two modes: different range and different wavelengths
        #mode = "_oxcco2_water"
        mode = "_all"

        if mode == "_oxcco1" or mode == "_oxcco2" or mode == "_oxcco2_water":
            left_cut = 740
            right_cut = 910
        elif mode == "_oxcco2_2":
            left_cut = 700
            right_cut = 910
        elif mode == "_oxcco2_3":
            left_cut = 740
            right_cut = 910
        elif mode == "_oxcco4":
            left_cut = 580
            right_cut = 910
        elif mode == "_no_water":
            left_cut = 500
            right_cut = 910 
        elif mode == "_all":
            left_cut = 500
            right_cut = 910         
        else:
            error("Nonexistent mode")
            
        molecules_cyto, x_cyto = read_molecules_cytochrome_cb(left_cut, right_cut, self.wavelengths)
        molecules, x = read_molecules_creatis(left_cut, right_cut, x_waves=self.wavelengths)
        y_c_oxy, y_c_red, y_b_oxy, y_b_red = molecules_cyto #, y_water, y_fat = molecules
        y_hb_f, y_hbo2_f, y_coxa, y_creda, y_fat, y_water = molecules

        if mode == "_oxcco1":
            M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                        np.asarray(y_hb_f),
                                        np.asarray(y_coxa),
                                        np.asarray(y_creda))))      
        elif mode == "_oxcco2":
            M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                        np.asarray(y_hb_f),
                                        np.asarray(y_coxa),
                                        np.asarray(y_creda),
                                        np.asarray(y_fat))))        
        elif mode == "_oxcco2_2":
            M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                        np.asarray(y_hb_f),
                                        np.asarray(y_coxa),
                                        np.asarray(y_creda),
                                        np.asarray(y_fat),
                                        np.asarray(y_water))))         
        elif mode == "_oxcco3" or mode == "_oxcco4" or mode == "_oxcco2_water":
            M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                        np.asarray(y_hb_f),
                                        np.asarray(y_coxa),
                                        np.asarray(y_creda),
                                        np.asarray(y_water),
                                        np.asarray(y_fat))))
        elif mode == "_no_water":
            M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                        np.asarray(y_hb_f),
                                        np.asarray(y_coxa),
                                        np.asarray(y_creda),
                                        np.asarray(y_c_oxy),
                                        np.asarray(y_c_red),
                                        np.asarray(y_b_oxy),
                                        np.asarray(y_b_red),
                                        np.asarray(y_fat))))        
        elif mode == "no_fat":
            M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                        np.asarray(y_hb_f),
                                        np.asarray(y_coxa),
                                        np.asarray(y_creda),
                                        np.asarray(y_c_oxy),
                                        np.asarray(y_c_red),
                                        np.asarray(y_b_oxy),
                                        np.asarray(y_b_red),
                                        np.asarray(y_water))))      
        elif mode=="" or mode == "_all":    
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
            
        return M, x

class optimizationDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def setup(self, spec: OperatorSpec):
        spec.input("img")
        spec.input("img_ref")
        spec.input("b")
        spec.input("M")
        spec.input("x")

        spec.output("errors")
        spec.output("coef_list")
        spec.output("scattering_params")
        spec.output("errors_scatter")
        spec.output("img_ref")
        spec.output("t1")

    def compute(self, op_input, op_output, context):
        img = op_input.receive("img")
        img_ref = op_input.receive("img_ref")
        b = op_input.receive("b")
        M = op_input.receive("M")
        x = op_input.receive("x")
        print("[optimizationDataOp]: Received data")

        t1 = torch.load("/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/hyperprobe_biopsies/t1.pt").numpy()
        print("[optimizationDataOp]: Loaded t1")
        print("[optimizationDataOp]: Optimization started")
        # errors, coef_list, scattering_params, errors_scatter = helicoid_optimisation_ti_parallel(t1, b, M, x)
        # coef_list = coef_list.reshape(img.shape[0], img.shape[1], -1)
        # scattering_params = scattering_params.reshape(img.shape[0], img.shape[1], -1)
        errors=1
        coef_list=1
        scattering_params=1
        errors_scatter=1

        # op_output.emit(rgb_data_normalized.astype(np.float32), "out")
        print("[optimizationDataOp]: Found optimal solution")

        op_output.emit(errors, "errors")
        op_output.emit(coef_list, "coef_list")
        op_output.emit(scattering_params, "scattering_params")
        op_output.emit(errors_scatter, "errors_scatter")
        op_output.emit(img_ref, "img_ref")
        op_output.emit(t1, "t1")
        print("[optimizationDataOp]: Sent data")
        
class visualizeDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # spec.input("segmentation")
        spec.input("errors")
        spec.input("coef_list")
        spec.input("scattering_params")
        spec.input("errors_scatter")
        spec.input("img_ref")
        spec.input("t1")
        
    def compute(self, op_input, op_output, context):
        # seg = op_input.receive("segmentation")
        errors = op_input.receive("errors")
        coef_list = op_input.receive("coef_list")
        scattering_params = op_input.receive("scattering_params")
        errors_scatter = op_input.receive("errors_scatter")
        img_ref = op_input.receive("img_ref")
        t1 = op_input.receive("t1")
        print("[visualizeDataOp]: Received data")

        # errors = np.array([])
        # errors = self.to_numpy_array(errors)

        time_taken = -1
        # save_optimization_data(img_ref, np.array([128,128]), coef_list, scattering_params, errors, t1, time_taken, biopsy_name + "_s1_normalized_t1_fixed")
        
class biopsyApp(Application):
    def compose(self):
        count = 1
        loadData = loadDataOp(self, CountCondition(self, count), name="loadData")
        preProcessData = preProcessDataOp(self, name="preprocessData")
        optimizeData = optimizationDataOp(self, name="inferenceData")
        visualizeData = visualizeDataOp(self, name="visualizeData")
        self.add_flow(preProcessData, optimizeData, {("img", "img"), ("b", "b"), ("M", "M"), ("x", "x"), ("img_ref", "img_ref")})
        self.add_flow(optimizeData, visualizeData, {("errors", "errors"), ("coef_list", "coef_list"), ("scattering_params", "scattering_params"), ("errors_scatter", "errors_scatter"), ("img_ref", "img_ref"), ("t1", "t1")})
        self.add_flow(loadData, preProcessData, {("biopsy_ref", "biopsy_ref"), ("biopsy", "biopsy")})

def main():
    app = biopsyApp()
    app.run()

if __name__ == "__main__":
    main()