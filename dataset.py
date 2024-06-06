
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Callable


class VisualOdometryDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:

        self.sequences = []

        directories = [d for d in os.listdir(
            dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in directories:

            aux_path = f"{dataset_path}/{subdir}"

            # read data
            rgb_paths = self.read_images_paths(aux_path)

            if not validation:
                ground_truth_data = self.read_ground_truth(aux_path)
                interpolated_ground_truth = self.interpolate_ground_truth(
                    rgb_paths, ground_truth_data)
                


            # TODO: create sequences
            for i in range(1, len(rgb_paths)-1, 2):

                if not validation:
                    pos_first_image = np.array(interpolated_ground_truth[i-1][1])
                    pos_second_image = np.array(interpolated_ground_truth[i][1])
                    difference = np.subtract(pos_second_image, pos_first_image)
                else:
                    difference = None

                self.sequences.append((rgb_paths[i][0], 
                                       rgb_paths[i][1], 
                                       rgb_paths[i+1][0], 
                                       rgb_paths[i+1][1], 
                                       difference)) 

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.TensorType:

        # Load sequence of images
        sequence_images = []
        # ground_truth_pos = []
        timestampt = 0

        _, path, timestamp, next_path, difference = self.sequences[idx]

        # Load images
        imagen = cv2.imread(path)
        next_imagen = cv2.imread(next_path)

        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        next_imagen = cv2.cvtColor(next_imagen, cv2.COLOR_BGR2RGB)

        imagen = self.transform(imagen)
        next_imagen = self.transform(next_imagen)

        sequence_images.append(imagen)
        sequence_images.append(next_imagen)

        
        timestampt = timestamp

        sequence_images = torch.stack(sequence_images)

        return (sequence_images, torch.Tensor([0]), timestampt) if difference is None else (sequence_images, torch.Tensor(difference), timestampt)

    def read_images_paths(self, dataset_path: str) -> Tuple[float, str]:

        paths = []

        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"

                paths.append((timestamp, image_path))

        return paths

    def read_ground_truth(self, dataset_path: str) -> Tuple[float, Tuple[float]]:

        ground_truth_data = []

        with open(f"{dataset_path}/groundtruth.txt", "r") as file:

            for line in file:

                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))

        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: Tuple[float, str],
            ground_truth_data: Tuple[float, Tuple[float]]
    ) -> Tuple[float, Tuple[float]]:

        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:

            nearest_idx = np.argmin(
                np.abs(np.array(ground_truth_timestamps) - rgb_timestamp))

            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
