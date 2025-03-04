# Face Recognition System

This repository contains the code for a Face Recognition System that allows for the collection, training, and
recognition of faces. The project uses Python, and the OpenCV library to implement the face recognition functionality.
It can detect and recognize multiple faces from live video input.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements a face recognition system using computer vision and machine learning. The system can collect
face data, train a model, and recognize faces in real-time using a camera.

The Face Recognition System provides a command-line tool for face collection, training, and recognition. The system
recognizes multiple persons, and its training dataset is structured in YOLO format. The facial recognition system can be
customized for specific use cases.

The project is designed to work with video input or image files, and it includes functionality for augmenting the
dataset with random transformations for improved training accuracy.

## Features

- **Data Collection**: Collect face images and organize them into a dataset.
- **Face Detection**: Detect faces in real-time using a YOLO model.
- **Face Recognition**: Recognize faces and classify them based on a trained model.
- **Text-to-Speech**: Provides audio feedback during data collection and recognition processes.
- **Train/Test Split**: Automatically splits collected data into training, validation, and test sets.

## Setup

### Prerequisites

- Python 3.x
- OpenCV
- PyTorch
- YOLOv8
- pyttsx3

### Requirements

- Python 3.9 or 3.10
- ultralytics
- pyttsx3
- argparse
- OpenCV
- NumPy

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/thippeswammy/FaceRecognition.git
   cd FaceRecognition
   ```

2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

### requirements.txt

```plaintext
opencv-python==4.5.5.64
ultralytics
pyttsx3
argparse
```
### File Structure

```bash
FaceRecognitionSystem/
│
├── CollectedDataset/              # Directory for collected face images
├── TrainingDataSetForClassification/ # Directory for training datasets
├── runs/                           # Directory for model runs and weights
├── Register.py                    # Main script for running the face recognition system
└── requirements.txt               # Required Python packages
```

## Usage

### Face Collection

To collect faces and organize them for training, run the following command:

```bash
python Register.py --persons <person1> <person2> --test-cases <dataset_directory>
```

This will collect images of the specified person and store them in the specified output directory.

### Example:

```bash
python Register.py --persons ajaySir thippeswamy --test-cases case1
```

This will collect face data for ajaySir and thippeswamy in the 'CollectedDataset/case1'.

### Training the Model

Once you have collected faces, you can train the face recognition model using:

```bash
python Register.py --train --test-cases <dataset_directory>
```

### Example:

```bash
python Register.py --train --test-cases case1
```

### Face Recognition

After training, run this command to start recognizing faces:

```bash
python Register.py --find --test-cases <dataset_directory>
```

## Contributing

If you'd like to contribute to this project, feel free to submit a pull request. For major changes, please open an issue
first to discuss what you'd like to change.

### Steps to Contribute


1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Push to your fork.
6. Create a pull request.

### Next plain

the input should support the images video, camera, streaming..
You can also use live video from your webcam by running:

```bash
python Register.py --input webcam
```