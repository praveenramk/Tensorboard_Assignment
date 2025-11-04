# TensorBoard Visualization Assignment

## Overview

This assignment demonstrates building a deep learning pipeline for digit classification using the MNIST dataset with comprehensive TensorBoard visualization. The project showcases data loading, preprocessing, CNN model building, training with TensorBoard logging, evaluation, and real-time metrics visualization.

## Key Features

- **MNIST Digit Classification**: Complete pipeline for handwritten digit recognition
- **TensorBoard Integration**: Real-time training metrics visualization
- **Convolutional Neural Network**: CNN architecture optimized for image classification
- **Interactive Visualization**: Live monitoring of training progress and model performance
- **Jupyter Notebook Implementation**: Step-by-step walkthrough with explanations

## Project Structure

```
Tensorboard_Assignment/
├── Tensorboard_Visualization.ipynb    # Main notebook with complete pipeline
├── visualization/                     # Directory for visualization assets
└── logs/                             # TensorBoard log files (generated during training)
```

## Assignment Objectives

Build a comprehensive deep learning pipeline that includes:

1. **Data Loading & Preprocessing**: Load MNIST dataset and normalize pixel values
2. **Model Architecture**: Design a CNN with convolutional, pooling, and dropout layers
3. **Training Pipeline**: Train the model with proper validation
4. **TensorBoard Logging**: Configure callbacks for metrics tracking
5. **Visualization**: Launch TensorBoard for real-time monitoring
6. **Performance Analysis**: Evaluate model accuracy and loss patterns

## Dataset

### MNIST Database
- **Total Samples**: 70,000 grayscale handwritten digit images
- **Training Set**: 60,000 images  
- **Test Set**: 10,000 images
- **Image Dimensions**: 28 × 28 pixels
- **Classes**: 10 (digits 0-9)
- **Pixel Values**: Normalized to [0, 1] range

## Model Architecture

### Convolutional Neural Network (CNN)

```python
Sequential([
    Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Conv2D(10, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(10, (3, 3), activation='relu'),
    Conv2D(10, (3, 3), activation='relu'),
    MaxPooling2D((1,1)),
    Flatten(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

**Architecture Details**:
- **Input Layer**: 28×28×1 grayscale images
- **Convolutional Layers**: 4 Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling for dimensionality reduction
- **Regularization**: Dropout layer (0.2) to prevent overfitting
- **Output Layer**: Dense layer with softmax for 10-class classification

## Getting Started

### Prerequisites

```bash
pip install tensorflow matplotlib jupyter tensorboard
```

### Running the Assignment

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook Tensorboard_Visualization.ipynb
   ```

2. **Execute Cells Sequentially**:
   - Run each cell in order to build the complete pipeline
   - The notebook includes data visualization and model training

3. **Launch TensorBoard** (from notebook):
   ```python
   %load_ext tensorboard
   %tensorboard --logdir=./logs
   ```

4. **Access TensorBoard**:
   - Open the provided URL in your browser
   - Monitor training metrics in real-time

## Pipeline Workflow

### 1. Data Loading & Preprocessing
- Load MNIST dataset using TensorFlow/Keras
- Normalize pixel values from [0, 255] to [0, 1]
- Visualize sample images from the training set

### 2. Model Building
- Define CNN architecture with multiple convolutional layers
- Add pooling layers for feature reduction
- Include dropout for regularization

### 3. Model Compilation
- **Optimizer**: Adam optimizer for efficient training
- **Loss Function**: Sparse categorical crossentropy
- **Metrics**: Accuracy for performance monitoring

### 4. TensorBoard Setup
- Configure TensorBoard callback with histogram logging
- Set up log directory for metrics storage
- Enable real-time visualization

### 5. Model Training
- Train for 5 epochs with batch size of 32
- Use validation data for performance monitoring
- Log metrics automatically to TensorBoard

### 6. Visualization & Analysis
- Launch TensorBoard for interactive visualization
- Monitor training/validation accuracy and loss
- Analyze model performance trends

## TensorBoard Features

### Available Visualizations

1. **Scalars**: Training and validation metrics over time
   - Accuracy curves
   - Loss curves
   - Learning rate schedules

2. **Histograms**: Weight and bias distributions
   - Layer-wise parameter analysis
   - Gradient flow visualization

3. **Images**: Sample predictions and filters
   - Input image samples
   - Learned feature maps

## Expected Results

- **Training Accuracy**: ~98-99% after 5 epochs
- **Validation Accuracy**: ~97-98% on test set
- **Quick Convergence**: Model learns efficiently with proper architecture
- **Visualization**: Clear trends in TensorBoard metrics

## Technical Requirements

- **Python**: 3.7+
- **TensorFlow**: 2.x
- **Jupyter**: For interactive development
- **TensorBoard**: For visualization
- **Matplotlib**: For plotting



