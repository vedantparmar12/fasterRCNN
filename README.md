# Kaggle Wheat Detection using Faster R-CNN ResNet50

This repository contains a PyTorch implementation of the Faster R-CNN object detection model with a ResNet50 backbone for detecting wheat heads in images from the Kaggle Global Wheat Detection Competition.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Faster R-CNN (Faster Region-based Convolutional Neural Network) is a popular object detection algorithm that combines a Region Proposal Network (RPN) and a Fast R-CNN detector in a single network. This project uses a ResNet50 backbone pre-trained on ImageNet as the feature extractor for the Faster R-CNN model.

## Dataset

The dataset used in this project is from the [Global Wheat Detection Challenge](https://www.kaggle.com/c/global-wheat-detection) on Kaggle. It consists of images of wheat fields with bounding box annotations for wheat heads. The dataset is split into training and test sets, and the training set is further divided into subsets for training and validation.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/vedantparmar12//kaggle_wheat_detection_faster_rcnn_resnet50.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the Kaggle dataset and extract it to the `data/` directory.

2. Preprocess the data:

```bash
python preprocess.py
```

3. Train the model:

```bash
python train.py
```

4. Evaluate the model:

```bash
python evaluate.py
```

5. Make predictions on the test set:

```bash
python predict.py
```

Refer to the individual script files for more details and command-line arguments.

## Results

The trained model achieves an mAP (mean Average Precision) of `0.68` on the validation set and `0.79` on the test set.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs, improvements, or new features.

## License

This project is licensed under the [MIT License](LICENSE).
