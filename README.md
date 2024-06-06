# vo_lstm

## Installation

```shell
$ pip3 install -r requirements.txt
```

## Dataset

The dataset employed in this Visual odometry demo is the [TUM dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Specifically, the following sequences are used to train the model:

- [freiburg2_pioneer_360](https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_360.tgz)
- [freiburg2_pioneer_slam](https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz)
- [freiburg2_pioneer_slam2](https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam2.tgz)
- [freiburg2_pioneer_slam3](https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam3.tgz)

You have to download and extract the each sequence into the dataset/train directory.

## Train

First, configure the params to train the model. Check them in the [params.py](params.py) file. Then, run the training script.

```shell
$ python3 train.py
```

## Validation

The [TUM online validation tool](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/online_evaluation) is used to validate the model. For this aim, the following sequences can be used. Download and extract this sequence into the dataset/val directory.

- [freiburg2_pioneer_360_validation](https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_360_validation.tgz)
- [freiburg1_room_validation](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room_validation.tgz)
- [freiburg3_walking_rpy_validation](https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_rpy_validation.tgz)

Before using the validation tool, generate the position and orientations for the validation sequence. Then, upload the generated file and configure the validation tool setting the sequence length that is the frames per pose.

```shell
$ python3 val.py
```

<img src="./docs/validation_tool.png" width="100%" />

## Results

Put your results here showing the graphs got from [TUM online validation tool](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/online_evaluation).

Datset utilizado para validación: rgbd_dataset_freiburg3_walking_rpy_validation
## Entrenamiento 1 

hidden_size = 1000
num_layers = 2
bidirectional = False
lstm_dropout = 0.3

sequence_length = 2
batch_size = 16

learning_rate = 0.001
epochs = 2

![entrenamiento1](https://github.com/ULE-MURIA-2023-2024/pr-vo-lstm-alexmarcos10/assets/61562132/853a693b-16de-4528-80d9-380b869f0720)

## Entrenamiento 2
hidden_size = 500
num_layers = 2
bidirectional = True
lstm_dropout = 0.3

sequence_length = 2
batch_size = 16

learning_rate = 0.001
epochs = 2

![entrenamiento2](https://github.com/ULE-MURIA-2023-2024/pr-vo-lstm-alexmarcos10/assets/61562132/c40b22a9-2d4b-4b74-afa6-10a43f2e708a)

## Entrenamiento 3
hidden_size = 1200
num_layers = 4
bidirectional = True
lstm_dropout = 0.2

sequence_length = 2
batch_size = 32

learning_rate = 0.001
epochs = 3

![entrenamiento3](https://github.com/ULE-MURIA-2023-2024/pr-vo-lstm-alexmarcos10/assets/61562132/b54f9c6f-9ca8-4cda-89cd-527103529f71)
