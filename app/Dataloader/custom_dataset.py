import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import scipy.io as sio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MPIIGaze(Dataset):
    def __init__(self, mpii_dir):
        """Initialize the dataset by processing the given data of the normalized folder

        Parameters
        ----------
        mpii_dir : str
            the path of the directory
        """
        super().__init__()
        self.mpii_dir = mpii_dir
        person_list = os.listdir(self.mpii_dir + "/Data/Normalized")
        days_list = os.listdir(self.mpii_dir + f"/Data//Normalized/{person_list[0]}")
        days_list.sort()
        days_list = [x.replace(".mat", "") for x in days_list[:7]]
        self.entries = [
            {"person": x, "day": y, "image_index": z, "side": side}
            for x in person_list
            for y in days_list
            for z in range(2)
            for side in ["left", "right"]
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self._sample_data(index)

    def _sample_data(self, index):
        """helper method for extracting data for the dataset

        Parameters
        ----------
        index : int

        Returns
        -------
        tuple of torch tensors
            the tensor consists of image array, theta and phi
        """
        entry = self.entries[index]
        mat_path = (
            self.mpii_dir
            + "/Data/Normalized/"
            + entry["person"]
            + "/"
            + entry["day"]
            + ".mat"
        )
        mat = sio.loadmat(mat_path)
        filename = mat["filenames"]
        row = entry["image_index"]
        if len(filename) <= row:
            row = np.random.randint(0, len(filename))
        side = entry["side"]
        image_array = mat["data"][side][0, 0]["image"][0, 0][row]
        image_array = cv2.equalizeHist(image_array)
        image_array = image_array / 255.0
        image_array = np.resize(image_array, (227, 227))
        image_array = image_array.astype("float32")
        
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        x, y, z = mat["data"][side][0, 0]["gaze"][0, 0][row]
        if side == "right":
            image_array = np.fliplr(image_array)
            theta = np.arcsin(y)
        else:
            theta = np.arcsin(-y)
        phi = np.arctan2(-x, -z)

        image_array = image_array.reshape((3, 227, 227))
        theta = np.array(np.degrees([theta]), dtype=np.float32)
        phi = np.array(np.degrees([phi]), dtype=np.float32)
        return (
            torch.from_numpy(image_array.copy()).to(device),
            torch.from_numpy(theta.copy()).to(device),
            torch.from_numpy(phi.copy()).to(device)
        )
