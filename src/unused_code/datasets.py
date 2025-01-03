from spectral import open_image
from torch.utils.data import Dataset
import torch
import os

def BilFile_to_Tensor(img):
    np_img = img.read_subimage(range(img.shape[0]), range(img.shape[1])).astype(float)
    tensor_img = torch.from_numpy(np_img).float()
    tensor_img = tensor_img.permute(2, 0, 1)
    return tensor_img


class HELICoiD(Dataset):
    def __init__(self, paths):
        for path in paths:
            hdr_path = "{}/raw.hdr".format(path)
            gt_path = "{}/gtMap.hdr".format(path)
            self.img = BilFile_to_Tensor(open_image(hdr_path))
            self.gt = BilFile_to_Tensor(open_image(gt_path))

        print(self.img.shape)
        self.dataset_size = 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.img, self.gt

def get_patient_list(path):
    patient_dirs = os.listdir(path)
    patient_dirs = [path + i for i in patient_dirs]
    dirs_with_gt = []
    for dir in patient_dirs:
        hdr_path = "{}/raw.hdr".format(dir)
        gt_path = "{}/gtMap.hdr".format(dir)
        if os.path.exists(hdr_path) and os.path.exists(gt_path):
            dirs_with_gt.append(dir)
    return dirs_with_gt
