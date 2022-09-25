import cv2
import json
from matplotlib.pyplot import hist
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import numpy as np
import processing
from torch.utils.data import random_split

bolognese1 = [
    {'title': 'recording_2022_05_25_11_55_19',
     'start': 7, 'end': 18, 'label': 0},  # 0 people
    {'title': 'recording_2022_05_25_11_46_26',
     'start': 7, 'end': 18, 'label': 1},  # 1 person
    {'title': 'recording_2022_05_25_11_47_20',
     'start': 7, 'end': 18, 'label': 2},  # 2 people
    {'title': 'recording_2022_05_25_11_48_56',
     'start': 7, 'end': 18, 'label': 3},  # 3 people
]

pesto1 = [
    {'title': 'recording_2022_05_25_19_06_11',
     'start': 2, 'end': 50, 'label': 0},  # 0 people
    {'title': 'recording_2022_05_25_19_07_53',
     'start': 1, 'end': 90, 'label': 1},  # 1 person
    {'title': 'recording_2022_05_25_19_10_22',
     'start': 1, 'end': 110, 'label': 2},  # 2 people
    {'title': 'recording_2022_05_25_19_13_55',
     'start': 4, 'end': 110, 'label': 3},  # 3 people
]

bolognese1EPesto1 = bolognese1 + pesto1

tonno = [
    {"title": "recording_2022_05_26_11_08_08", 'start': 0, 'end': 4.5, 'label': 0},
    {"title": "recording_2022_05_26_11_08_08",
        'start': 4.5, 'end': 62, 'label': 1},
    {"title": "recording_2022_05_26_11_08_08",
        'start': 62, 'end': 98, 'label': 2},
    {"title": "recording_2022_05_26_11_08_08",
        'start': 98, 'end': 109, 'label': 1},
    {"title": "recording_2022_05_26_11_08_08",
        'start': 109, 'end': 135, 'label': 2},
    {"title": "recording_2022_05_26_11_08_08",
        'start': 135, 'end': 150, 'label': 0},
    # {"title": "recording_2022_05_26_11_08_08",
    #     'start': 150, 'end': 188, 'label': 1},
    {"title": "recording_2022_05_26_11_08_08",
        'start': 188, 'end': 192, 'label': 0},
    # {"title": "recording_2022_05_26_11_08_08",
    #     'start': 192, 'end': 200, 'label': 1},
    # {"title": "recording_2022_05_26_11_08_08",
    #     'start': 200, 'end': 209, 'label': 0},
    # {"title": "recording_2022_05_26_11_08_08",
    #     'start': 209, 'end': 215, 'label': 1},
    # {"title": "recording_2022_05_26_11_08_08",
    #     'start': 215, 'end': 225, 'label': 0},
    # {"title": "recording_2022_05_26_11_08_08",
    #     'start': 225, 'end': 239, 'label': 1}
]

class SensorDataset(Dataset):
    def __init__(self, recording_list):
        self.samples = []

        for recording in recording_list:
            file_name = f'../recordings/{recording["title"]}/RadarIfxAvian_00/radar.npy'
            data = np.load(file_name)
            start_frame, end_frame, labelix = recording["start"], recording["end"], recording["label"]

            label = torch.tensor([0, 0, 0, 0])
            label[labelix] = 1

            processed_data = processing.processing_rangeDopplerData(data[start_frame:end_frame])

            processed_data = preprocess_input(processed_data)

            # iterate frames
            for f in processed_data:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SensorCameraDataset(Dataset):
    def __init__(self, recording_list):
        self.samples = []

        for recording in recording_list:
            file_name = f'../recordings/{recording["title"]}/RadarIfxAvian_00/radar.npy'
            video_file = f'../recordings/{recording["title"]}/CamOpenCV_00/rgb.mp4'
            data = np.load(file_name)
            cap = cv2.VideoCapture(video_file)

            nOfDataFrames = data.shape[0]
            nOfVideoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            def get_frame(video_frame):
                k = nOfDataFrames/nOfVideoFrames
                return int(video_frame * k)

            start_frame, end_frame, labelix = get_frame(
                recording["start"]*30), get_frame(recording["end"]*30), recording["label"]

            label = torch.tensor([0, 0, 0, 0])
            label[labelix] = 1

            processed_data = processing.processing_rangeDopplerData(
                data[start_frame:end_frame])

            processed_data = preprocess_input(processed_data)

            # iterate frames
            for f in processed_data:

                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def preprocess_input(processed_data, skip_gaussian=False):
    processed_data = np.abs(processed_data)

    processed_data[:, :, 32:34, :] = 0  # Remove the middle band (noise?)

    # for i in range(len(processed_data)):
    #     for j in range(len(processed_data[i])):
    # processed_data[i, j] -= processed_data[i, j].mean()  # subtract the mean
            # processed_data[i, j] = gaussian_filter(processed_data[i, j], (1, 1))  # Gaussian blur

    processed_data[:, :, :, 45:] = 0  # Remove far away objects
    processed_data *= 1.0 / processed_data.max()  # Normalize
    return processed_data


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':
    dataset = SensorDataset()
    train_size = round(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    # torch.save(dataset, 'datasets/bolognese1.pth')
    torch.save(dataset, 'datasets/pesto1.pth')
