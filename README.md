# Image Classification Using Convolutional Neural Network (CNN)

## Overview
This project demonstrates image classification using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The 10 classes represent:
- Airplanes
- Automobiles
- Birds
- Cats
- Deer
- Dogs
- Frogs
- Horses
- Ships
- Trucks

## Key Features
- Implementation of both simple Artificial Neural Network (ANN) and CNN models
- Data preprocessing including normalization
- Visualization of sample images from the dataset
- Model evaluation with accuracy metrics
- Implementation of an improved CNN model with techniques like:
  - Batch normalization  
  - Dropout layers  
  - L2 regularization  
  - Learning rate scheduling  
  - Data augmentation

## Model Performance
- Simple ANN: ~49% accuracy  
- Basic CNN: ~70% accuracy  
- Improved CNN with advanced techniques: **89.10%** test accuracy

## About VGGNet (Very Deep Convolutional Networks)
This project draws inspiration from the seminal paper ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) by Karen Simonyan and Andrew Zisserman. The paper introduced the VGG network architecture which demonstrated that increasing network depth (using small 3×3 convolution filters) leads to significant improvements in accuracy.

### Key aspects of VGGNet implemented in this project:
- Use of multiple convolutional layers with small receptive fields (3×3)
- Stacking convolutional layers before pooling operations
- Increasing depth of the network while maintaining simplicity in architecture

The improved CNN model in this notebook incorporates several VGG-inspired modifications:
- Deeper architecture with multiple convolutional blocks
- Batch normalization after convolutional layers
- Regularization techniques to prevent overfitting

## Requirements
- Python 3  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib  

## Usage
Run the notebook cells sequentially to:
1. Load and preprocess the CIFAR-10 dataset  
2. Visualize sample images  
3. Train and evaluate the ANN and CNN models  
4. Implement and test the improved CNN model  

The notebook includes comments explaining each step of the process.

## Results
The final improved CNN model achieves significantly better performance than the initial implementations, demonstrating the effectiveness of deeper architectures and regularization techniques for image classification tasks.
