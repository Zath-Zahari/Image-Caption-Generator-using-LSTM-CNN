# Image Caption Generator Using CNN-LSTM

Welcome to the **EC9170 Mini Project 2025** repository from the University of Jaffna! This project implements an image caption generator built from scratch, using a custom Convolutional Neural Network (CNN) for feature extraction and a Long Short-Term Memory (LSTM) network for generating natural language captions. 

## Project Overview
The model takes an image as input and generates a descriptive caption by combining computer vision and natural language processing. Built without pre-trained models (e.g., VGG16), it uses a custom CNN to extract visual features and an LSTM to generate word-by-word captions. Trained on 8,091 images with 48,455 human-annotated captions, it produces grammatically correct and contextually relevant results. 🖼️📝

- **Group Members**: 2021/E/010, 2021/E/146, 2021/E/185
- **Institution**: University of Jaffna
- **Date**: April 2025

## System Architecture
- **CNN**: Custom-built with convolutional layers, ReLU activation, and max-pooling to extract a 256-unit feature vector from 224x224 images. 
- **LSTM**: Processes image features and tokenized captions to generate coherent sentences. 
- **Dataset**: 8,091 images with 48,455 cleaned captions, preprocessed with tokenization and padding. 
- **Training**: 10 epochs with Adam optimizer (learning rate 0.001), categorical cross-entropy loss, and early stopping. 
- **Platform**: Developed in Python using Keras, trained on Google Colab (GPU). 

## How It Works
1. **Image Preprocessing**: Resize to 224x224, normalize pixels to [0,1]. 
2. **Feature Extraction**: Custom CNN extracts a feature vector from the image. 
3. **Caption Generation**: LSTM takes the feature vector and generates a sequence of words, using `<start>` and `<end>` tokens. 
4. **Output**: A complete sentence describing the image (e.g., "A man lays on a bench with his dog"). 


## Challenges Faced
- **Limited Compute Resources**: Google Colab runtime limits restricted training to 10 epochs. 
- **Slow Feature Extraction**: Custom CNN processing of 8,091 images was time-intensive.
- **Memory Constraints**: Large dataset and model size required small batch sizes (32).
- **Caption Quality**: Limited epochs led to occasional vague or repetitive captions.

## Results
- **Quantitative**: Training/validation loss curves showed convergence, though limited by 10 epochs.
- **Qualitative**: Generated grammatically correct captions with solid object recognition (e.g., "A child in a pink dress climbs stairs").
- **Limitations**: Complex scenes sometimes produced vague captions; more epochs could improve accuracy. 🔧

## How to Run
1. **Requirements**: Python, Keras, TensorFlow, NumPy, Pillow. Install via:
   ```bash
   pip install tensorflow numpy pillow

Setup: Clone this repository and place dataset images in data/. Download large files via Git LFS or Google Drive.

Run: Execute image_captioning.ipynb in Jupyter or Google Colab, or run scripts:
python code/cnn_model.py
python code/lstm_caption.py

Test: Use data/sample_image.jpg to generate captions.


**Future Improvements**
* Use pre-trained CNNs (e.g., VGG16) for faster feature extraction.
* Train for more epochs with better compute resources.
* Add attention mechanisms for focused captioning. 
* Implement BLEU/METEOR metrics for evaluation.
* Deploy as a web/mobile app for real-time captioning.

**Contributing**
Contributions are welcome! Fork this repository, make improvements, and submit pull requests.

**License**

This project is licensed under the MIT License. See LICENSE for details.

