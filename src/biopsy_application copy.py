import sys
sys.path.append("/workspace/holohub/benchmarks/holoscan_flow_benchmarking")
from benchmarked_application import BenchmarkedApplication

import scipy
import os
import h5py
import numpy as np
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

def init_interactive_visualization(image_tensor, wavelengths):
    def find_rgb_indices(wavelengths, rgb_wavelengths=[630, 540, 510]):
        rgb_indices = []
        for wl in rgb_wavelengths:
            index = np.abs(wavelengths - wl).argmin()
            rgb_indices.append(index)
        return rgb_indices
    
    rgb_indices = find_rgb_indices(wavelengths)
    
    def generate_rgb_image(image_tensor, rgb_indices):
        rgb_image = np.zeros((image_tensor.shape[0], image_tensor.shape[1], 3))
        for i, idx in enumerate(rgb_indices):
            rgb_image[:, :, i] = normalize(image_tensor[:, :, idx])
        rgb_image = rgb_image
        return rgb_image
    
    rgb_image = generate_rgb_image(image_tensor, rgb_indices)
    
    colors = list(mcolors.TABLEAU_COLORS.values()) 
    color_index = 0  
    
    def onclick(event):
        nonlocal color_index
        ix, iy = int(event.xdata), int(event.ydata)
        spectrum = image_tensor[iy, ix, :]
        
        ax1.plot(ix, iy, 'x', color=colors[color_index % len(colors)], markersize=10)
        ax2.plot(wavelengths, spectrum, color=colors[color_index % len(colors)], label=f'Pixel ({ix}, {iy})')
        ax2.set_title('Spectral Signatures')
        ax2.set_xlabel('Wavelength')
        ax2.set_ylabel('Intensity')
        ax2.legend()
        
        fig.canvas.draw()
        
        color_index += 1  
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(rgb_image)
    ax1.set_title('RGB Representation of Hyperspectral Image')
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show()

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
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        biopsy_name = "Biopsy_S1_reflectance"
        filename = "/workspace/media/m2/data/src/dataset/hyperprobe_biopsies/" + biopsy_name + ".mat"
        with h5py.File(filename, 'r') as f:
            data = np.array(f['Ref_hyper'])
        print("sending loaded data")
        op_output.emit(data, "out")

class processDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        print("received data in process op")
        data_transposed = np.transpose(data)
        data_transposed.shape
        data_transposed[data_transposed<=0] = 10**-3
        data_transposed = F.avg_pool2d(torch.tensor(data_transposed).permute(2, 0, 1).unsqueeze(0), kernel_size=4, stride=4).squeeze(0).permute(1, 2, 0).numpy()
        img = data_transposed
        data_transposed.shape
        
        print("finished processing, sending to viz")
        op_output.emit(data_transposed, "out")

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

        for i in range(len(wavelengths)):
            plt.figure()
            plt.imshow(data_transposed[:,:,i])
            plt.colorbar()
            plt.title("Wavelength " + str(wavelengths[i]))
            break

        semisynthetic_rgb = np.concatenate((normalize(data_transposed[:,:,23][:,:,np.newaxis]), normalize(data_transposed[:,:,6][:,:,np.newaxis]), normalize(data_transposed[:,:,0][:,:,np.newaxis])),axis=2)
        plt.imshow(semisynthetic_rgb)

        init_interactive_visualization(data_transposed[150:350, 150:350, :], wavelengths)
        print("reached end of application")

class biopsyApp(Application):
    def compose(self):
        loadData = loadDataOp(self, name="loadData")
        processData = processDataOp(self, name="processData")
        visualizeData = visualizeDataOp(self, name="visualizeData")

        self.add_flow(loadData,processData)
        self.add_flow(processData, visualizeData)

def main():
    app = biopsyApp()
    app.run()


if __name__ == "__main__":
    main()