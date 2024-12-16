import sys
sys.path.append("/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src")
sys.path.append('../../')

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
from optimisation_helicoid import read_molecules_creatis, read_molecules_cytochrome_cb, helicoid_optimisation_ti_parallel, helicoid_optimisation_scattering
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.conditions import CountCondition, PeriodicCondition


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Define the visualization function
# def visualize_hyperspectral_image(image_tensor, wavelength_index, wavelengths):
#     assert len(image_tensor.shape) == 3, "Image tensor must be 3-dimensional"
#     A, B, C = image_tensor.shape
#     assert 0 <= wavelength_index < C, "Wavelength index out of bounds"

#     variation = np.var(image_tensor[:,:,0:40], axis=2)
#     flat_indices = np.argpartition(variation.flatten(), -3)[-3:]
#     top_var_pixel_indices = np.array(np.unravel_index(flat_indices, variation.shape)).T

#     plt.figure(figsize=(14, 6))

#     plt.subplot(1, 2, 1)
#     plt.imshow(image_tensor[:, :, wavelength_index], cmap='gray')
#     plt.title(f'Hyperspectral Image at Wavelength {wavelength_index}')
#     plt.colorbar(label='Intensity')
    
#     colors = ['r', 'g', 'b']
#     for i, (x, y) in enumerate(top_var_pixel_indices):
#         plt.scatter(y, x, color=colors[i], s=100, marker='x', label=f'Pixel {i+1}')

#     plt.legend()

#     plt.subplot(1, 2, 2)
#     for i, (x, y) in enumerate(top_var_pixel_indices):
#         plt.plot(wavelengths, image_tensor[x, y, :], color=colors[i], label=f'Pixel {x},{y}')

#     plt.title('Spectral Signatures')
#     plt.xlabel('Wavelength')
#     plt.ylabel('Intensity')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

def save_optimization_data(img_ref, reference_point, coef_list, scattering_params, errors, t1, time_taken, folder):
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
    
    path = "/workspace/media/m2/data/src/dataset/hyperprobe_biopsies"
    
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

class loadDataOp(Operator):
    def __init__(self, *args, **kwargs):
        # condition = PeriodicCondition(self, 1000)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        print("loading biopsy probe file")
        ref_biopsy_name = "Biopsy_S1_reflectance"
        filename = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/hyperprobe_biopsies/" + ref_biopsy_name + ".mat"
        with h5py.File(filename, 'r') as f:
            data = np.array(f['Ref_hyper'])
        data_transposed = np.transpose(data)
        data_transposed.shape
        data_transposed[data_transposed<=0] = 10**-3
        data_transposed = F.avg_pool2d(torch.tensor(data_transposed).permute(2, 0, 1).unsqueeze(0), kernel_size=8, stride=8).squeeze(0).permute(1, 2, 0).numpy()

        mean_biopsy_s1 = np.mean(data_transposed[150:350, 150:350, :].reshape(-1, data_transposed.shape[-1]), axis=0)
        
        for biopsy_name in [("Biopsy_S2_reflectance","4")]:#, 
                    #("Biopsy_S4_fov1_reflectance", "4"), 
                    #("Biopsy_S4_fov2_reflectance", "4"),
                    #("Biopsy_S5_reflectance", "4"),
                    #("Biopsy_S7_reflectance", "2"),
                    #("Biopsy_S8_reflectance", "3"),
                    #("Biopsy_S9_reflectance", "4"),
                    #("Biopsy_S10_reflectance", "2"),
                    #("Biopsy_S11_reflectance","4")]:
            filename = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/hyperprobe_biopsies/" + biopsy_name[0] + ".mat"
            
            with h5py.File(filename, 'r') as f:
                data = np.array(f['Ref_hyper'])
            data_transposed = np.transpose(data)
            data_transposed.shape
            data_transposed[data_transposed<=0] = 10**-3
            data_transposed = F.avg_pool2d(torch.tensor(data_transposed).permute(2, 0, 1).unsqueeze(0), kernel_size=4, stride=4).squeeze(0).permute(1, 2, 0).numpy()

        print("sending loaded data")
        op_output.emit(mean_biopsy_s1, "ref_out")
        op_output.emit(data_transposed, "main_out")

class processDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("ref_in")
        spec.input("main_in")
        # spec.output("out")

    def compute(self, op_input, op_output, context):
        ref_biopsy_s1_mean = op_input.receive("ref_in")
        main_biopsy_data = op_input.receive("main_in") #data to be evaluated??
        print("received data in process op")

        wavelengths = np.linspace(510,900,79)
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
            
        molecules_cyto, x_cyto = read_molecules_cytochrome_cb(left_cut, right_cut, wavelengths)
        molecules, x = read_molecules_creatis(left_cut, right_cut, x_waves=wavelengths)
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
        else:
            error("Mode nonexistent")

        img = main_biopsy_data
        time_taken = -1
        coarseness=1

        img_ref = -np.log(img[::coarseness,::coarseness,:] / ref_biopsy_s1_mean)[:,:,np.in1d(self.wavelengths,x)]
        b = img_ref.reshape(-1, img_ref.shape[-1])
        errors, coef_list, scattering_params, errors_scatter = helicoid_optimisation_ti_parallel(t1, b, M, x)
        coef_list = coef_list.reshape(img.shape[0], img.shape[1], -1)
        scattering_params = scattering_params.reshape(img.shape[0], img.shape[1], -1)
        
        errors = np.array([])
        # coef_list, scattering_params, errors, a_t1, b_t1, img_ref, time_taken = helicoid_optimisation_scattering(img, None, M, x, coarseness=1, wavelength=wavelengths, reference_signal=mean_biopsy_s1)
        # save_optimization_data(img_ref, np.array([128,128]), coef_list, scattering_params, errors, t1, time_taken, biopsy_name[0]+"_s1_normalized_" + str(right_cut) + "_t1_fixed" + mode)
        
        
        # Emit this data
        # op_output.emit(rgb_data_normalized.astype(np.float32), "out")

        # op_output.emit(data_transposed, "out")

class visualizeDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        print("received in viz")
        data_transposed = op_input.receive("in")

        wavelengths = np.linspace(510,900,79)
        print(self.wavelengths)

        # for i in range(len(wavelengths)):
        #     plt.figure()
        #     plt.imshow(data_transposed[:,:,i])
        #     plt.colorbar()
        #     plt.title("Wavelength " + str(wavelengths[i]))
        #     break

        semisynthetic_rgb = np.concatenate((normalize(data_transposed[:,:,23][:,:,np.newaxis]), normalize(data_transposed[:,:,6][:,:,np.newaxis]), normalize(data_transposed[:,:,0][:,:,np.newaxis])),axis=2)
        # plt.imshow(semisynthetic_rgb)

        # init_interactive_visualization(data_transposed[150:350, 150:350, :], wavelengths)
        print("reached end of application")

class biopsyApp(Application):
    def __init__(self, data=None, output_folder=None):
        super().__init__()
        
    def compose(self):
        loadData = loadDataOp(self, name="loadData", condition=CountCondition(self, count=2))
        processData = processDataOp(self, time_taken=self.time_taken, name="processData")
        visualizeData = visualizeDataOp(self, name="visualizeData")
        viz = HyperspectralVizOp(self, name="viz")

        self.add_flow(loadData,processData)
        self.add_flow(processData, viz, {("out", "rgb")})
         

def main():
    app = biopsyApp()
    app.run()


if __name__ == "__main__":
    # config_file = os.path.join(os.path.dirname(__file__), "holoviz_config.yaml") #==/workspace/volumes/m2/data/src/file.yaml
    main()