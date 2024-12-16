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
from optimisation_helicoid import read_molecules_creatis, read_molecules_cytochrome_cb
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

# def save_optimization_data(img_ref, reference_point, coef_list, scattering_params, errors, t1, time_taken, folder):
#     """
#     Used after running the optimisation script/notebook to be used later by the Neural Network
#     """
#     img_ref = torch.from_numpy(img_ref)
#     reference_point = torch.from_numpy(reference_point)
#     coef_list = torch.from_numpy(coef_list)
#     scattering_params = torch.from_numpy(scattering_params)
#     errors = torch.from_numpy(errors)
#     t1 = torch.from_numpy(t1)
#     time_taken = torch.from_numpy(np.array([time_taken]))
    
#     path = "/workspace/media/m2/data/src/dataset/hyperprobe_biopsies"
    
#     if not os.path.exists(path): os.mkdir(path)
#     path = path + folder 
#     if not os.path.exists(path): os.mkdir(path)
    
#     torch.save(img_ref, path+'/img_ref.pt')
#     torch.save(reference_point, path+'/reference_point.pt')
#     torch.save(coef_list, path + '/coef_list.pt')
#     torch.save(scattering_params, path + '/scattering_params.pt')
#     torch.save(errors, path + '/errors.pt')
#     torch.save(t1, path + '/t1.pt')
#     torch.save(time_taken, path + '/time_taken.pt')

    # Visualize Wavelengths

    #filename = "/home/kevin/hyperprobe_biopsies/Biopsy_S2_reflectance.mat"

class loadDataOp(Operator):
    def __init__(self, *args, **kwargs):
        # condition = PeriodicCondition(self, 1000)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        print("loading biopsy probe file")
        biopsy_name = "Biopsy_S1_reflectance"
        filename = "/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src/dataset/hyperprobe_biopsies/" + biopsy_name + ".mat"
        with h5py.File(filename, 'r') as f:
            data = np.array(f['Ref_hyper'])
        print("sending loaded data")
        op_output.emit(data, "out")

def normalize(x):
    """Normalize data to [0, 1]."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class processDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        print("received data in process op")

        #convert cube
        data_transposed = np.transpose(data)
        data_transposed.shape
        data_transposed[data_transposed<=0] = 10**-3
        data_transposed = F.avg_pool2d(torch.tensor(data_transposed).permute(2, 0, 1).unsqueeze(0), kernel_size=4, stride=4).squeeze(0).permute(1, 2, 0).numpy()
        img = data_transposed
        data_transposed.shape

        #construct rgb
        semisynthetic_rgb = np.concatenate((normalize(data_transposed[:,:,23][:,:,np.newaxis]), normalize(data_transposed[:,:,6][:,:,np.newaxis]), normalize(data_transposed[:,:,0][:,:,np.newaxis])),axis=2)
        print(semisynthetic_rgb.shape)
        rgb_channels = [23, 6, 0]  # Replace these with the desired indices
        rgb_data = np.stack([data_transposed[:, :, i] for i in rgb_channels], axis=-1)

        # Normalize data to [0, 1]
        rgb_data_normalized = normalize(rgb_data) #necessary?
        print("finished processing, sending to viz")
        # Emit this data
        op_output.emit(rgb_data_normalized.astype(np.float32), "out")

        
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
        print(wavelengths)

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

class HyperspectralVizOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # spec.input("segmentation")
        spec.input("rgb")

    def compute(self, op_input, op_output, context):
        # seg = op_input.receive("segmentation")
        rgb = op_input.receive("rgb")



        #save array to not do the timeconsuming loading each time
        # reshaping the array from 3D matrice to 2D matrice.
        rgb_reshaped = rgb.reshape(rgb.shape[0], -1)

        # saving reshaped array to file.
        np.savetxt("geekfile.txt", rgb_reshaped)

        # retrieving data from file.
        # loaded_rgb = np.loadtxt("geekfile.txt")

        # load_original_rgb = loaded_rgb.reshape(
        #     loaded_rgb.shape[0], loaded_rgb.shape[1] // 3, 3)
        


        # rgb_seg = LABEL_COLORMAP[seg]

        # plt.figure(figsize=(18, 7))
        # plt.subplot(1, 3, 1)
        # plt.imshow(rgb)
        # plt.gca().set_title("RGB image")
        # plt.axis("off")
        # plt.savefig("test_plot.png")
        # plt.show()

class biopsyApp(Application):
    def compose(self):
        loadData = loadDataOp(self, name="loadData", condition=CountCondition(self, count=2))
        processData = processDataOp(self, name="processData")
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