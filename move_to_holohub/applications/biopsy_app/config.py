import os

# Please input the path to the Helicoid dataset
#path_helicoid = "/home/kevinscibilia/helicoid/HSI_Human_Brain_Database_IEEE_Access"

left_cut = 740
#left_cut = 780
right_cut = 900
molecule_count = 3
max_b = 5
max_a = 100
scattering_model = True
train_on_synthetic = True

# path variables
spectra_path = "/dataset/UCL-NIR-Spectra/spectra/"
dataset_path = "./dataset/"
path = os.getcwd()
print("PATH: ", path)

path_helicoid = "/home/home/ivan/HELICoiD/HSI_Human_Brain_Database_IEEE_Access"
path_absorp = "./dataset/UCL-NIR-Spectra/spectra/"
path_creatis = "./dataset/CREATIS-Spectra/spectra/"

left_cut_helicoid = 530
#left_cut_helicoid = 500
#right_cut_helicoid = 800
right_cut_helicoid = 750
# train variables
epochs = 1000
lr = 0.001
patience = 10