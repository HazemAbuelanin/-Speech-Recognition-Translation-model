# Speech-Recognition-Translation-model
This project implements a complete pipeline for speech recognition and translation using state-of-the-art machine learning models. The goal is to convert audio speech into text and subsequently translate it into another language.

Key Features:

    Speech-to-Text Conversion: Leveraging advanced speech recognition models to accurately transcribe spoken language into text.
    Text Translation: High-quality translation of transcribed text from the source language to the target language using a pretrained translation model.
    Hugging Face Deployment: The entire pipeline is deployed on Hugging Face, providing an API and web interface for easy interaction and real-time inference.

Project Objectives:

    Provide a robust solution for speech recognition and translation, facilitating multilingual applications.
    Simplify access to state-of-the-art models through Hugging Face integration.
    Enable seamless real-time interaction with the model via a web interface or API.
This repository contains a Jupyter Notebook that implements a speech recognition and translation pipeline. The project focuses on recognizing spoken language and translating it into another language using state-of-the-art machine learning models and techniques.

# Overview

The notebook demonstrates the following key steps:

    Speech Recognition: Converts spoken language into text.
    Translation: Translates the recognized text into the target language.
    Model Deployment: The trained model is deployed using Hugging Face for easy access and use.

## Techniques Used

### 🎛️ Preprocessing

- **Audio Processing**: 
  - The raw audio data is preprocessed, including:
    - Resampling audio signals to match the input requirements of the speech recognition model.
    - Converting audio files into the appropriate format using libraries such as `Librosa`.
  
- **Text Normalization**: 
  - Text normalization techniques are applied to handle special characters, punctuation, and case sensitivity for better translation quality.

### 🗣️ Speech Recognition Model

- **Pretrained Model**: 
  - A pretrained model from Hugging Face's `transformers` library is used for speech recognition. The model transforms the input audio into text.
  
- **Fine-Tuning**: 
  - Additional fine-tuning of the speech recognition model can be done on a custom dataset to enhance performance (if applicable).

### 🌐 Translation Model

- **Machine Translation**: 
  - A machine translation model is employed to convert the recognized text into the target language.

- **Language Pair**: 
  - The translation is performed between a source language (e.g., English) and a target language (e.g., French, Spanish, etc.).

### 🚀 Model Deployment

- **Hugging Face Hub**: 
  - The model is deployed using the **Hugging Face Hub**, providing an easy-to-use interface for inference. Hugging Face's API allows users to interact with the model directly in production environments.
  
- **Hugging Face Spaces**: 
  - The deployment process leverages **Hugging Face Spaces**, enabling users to test the model via a web interface.

## How to Use the Notebook

1. Clone this repository:

    ```bash
    git clone https://github.com/HazemAbuelanin/Speech-Recognition-Translation-model.git
    cd speech-recognition-translation
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook using Jupyter:

    ```bash
    jupyter notebook speech-recognition-translation.ipynb
    ```
4. Run the notebook using Jupyter:

    ```bash
    jupyter notebook pretrained-speechrecognition.ipynb
    ```

## Model Deployment on Hugging Face

The model is deployed on Hugging Face for easy use:

- **Hugging Face Hub**: The model is published on Hugging Face Hub, making it accessible for inference. Users can send requests to the model and get predictions.
  
- **Inference API**: The model is available via Hugging Face's Inference API, allowing integration into various applications.

- **Link to Deployed Model**: [Speech-recognition-translation](https://huggingface.co/spaces/FarahMohsenSamy1/ASR-Translation)

- **Link to Deployed pretrained Model**: [Speech-recognition-translation-pretrained](https://huggingface.co/spaces/FarahMohsenSamy1/ASR)

# Dependencies
- Dependencies are listed in the requirements.txt file. 
