import torch
from torch.utils.data import Dataset
import numpy as np
import processing
from torch.utils.data import random_split

bolognese1 = [
    {'title': 'recording_2022_05_25_11_55_19',
        'start': 200, 'end': 500, 'label': 0},  # 0 people
    {'title': 'recording_2022_05_25_11_46_26',
        'start': 200, 'end': 500, 'label': 1},  # 1 person
    {'title': 'recording_2022_05_25_11_47_20',
        'start': 200, 'end': 500, 'label': 2},  # 2 people
    {'title': 'recording_2022_05_25_11_48_56',
        'start': 200, 'end': 500, 'label': 3},  # 3 people
]

pesto1 = [
    {'title': 'recording_2022_05_25_19_06_11',
        'start': 50, 'end': 1500, 'label': 0},  # 0 people
    {'title': 'recording_2022_05_25_19_07_53',
        'start': 20, 'end': 2700, 'label': 1},  # 1 person
    {'title': 'recording_2022_05_25_19_10_22',
        'start': 20, 'end': 3300, 'label': 2},  # 2 people
    {'title': 'recording_2022_05_25_19_13_55',
        'start': 110, 'end': 3300, 'label': 3},  # 3 people
]

bolognese1EPesto1 = bolognese1 + pesto1

class SensorDataset(Dataset):
    def __init__(self, recording_list):
        self.samples = []

        for recording in recording_list:
            file_name = f'../../recordings/{recording["title"]}/RadarIfxAvian_00/radar.npy'
            data = np.load(file_name)
            start_frame, end_frame, labelix = recording["start"], recording["end"], recording["label"]

            label = torch.tensor([0, 0, 0, 0])
            label[labelix] = 1

            processed_data = processing.processing_rangeDopplerData(
                data[start_frame:end_frame])
            processed_data = np.abs(processed_data)

            processed_data[:, :, 32:34, :] = 0  # Take out high strip
            processed_data *= 1.0 / processed_data.max()  # Normalize

            # iterate frames
            for f in processed_data:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    dataset = SensorDataset()
    train_size = round(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    # torch.save(dataset, 'datasets/bolognese1.pth')
    torch.save(dataset, 'datasets/pesto1.pth')
