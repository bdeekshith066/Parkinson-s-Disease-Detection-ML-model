# Parkinson's Disease Prediction Machine Learning Model

This repository houses a machine learning model for predicting the presence of Parkinson's disease in individuals based on extensive voice-related features. The model, trained on a dataset with 195 entries and 24 columns, demonstrates a high accuracy of 88.462%.

## Overview

Parkinson's disease is a neurodegenerative disorder, and early prediction is crucial for effective intervention. This machine learning model utilizes voice-related features such as frequency measures, jitter, shimmer, and other acoustic parameters to accurately distinguish between individuals with Parkinson's disease and healthy subjects.

## Features

- **Machine Learning Algorithm**: The model employs a support vector machine (SVM) algorithm for precise prediction.

- **Accuracy**: Achieving an accuracy of 88.462%, this model provides reliable predictions for identifying individuals with Parkinson's disease.

- **Input Features**: The model considers a comprehensive set of 24 voice-related features, ensuring a robust analysis for accurate predictions.

## Data

### Columns

1. **name**: Patient's name
2. **MDVP:Fo(Hz)**: Average vocal fundamental frequency
3. **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency
4. **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency
5. **MDVP:Jitter(%)**: Frequency variation in the voice
6. **MDVP:Jitter(Abs)**: Absolute jitter in the voice
7. **MDVP:RAP**: Relative amplitude perturbation
8. **MDVP:PPQ**: Five-point period perturbation quotient
9. **Jitter:DDP**: Average absolute difference of differences between jitter cycles
10. **MDVP:Shimmer**: Amplitude variation in the voice
11. **MDVP:Shimmer(dB)**: Shimmer in decibels
12. **Shimmer:APQ3**: Amplitude perturbation quotient, measures variation in voice amplitude
13. **Shimmer:APQ5**: Amplitude perturbation quotient, measures variation in voice amplitude
14. **MDVP:APQ**:  Amplitude perturbation quotient
15. **Shimmer:DDA**: Three-point amplitude perturbation quotient
16. **NHR**: Noise-to-harmonics ratio
17. **HNR**: Harmonic-to-noise ratio
18. **status**: Parkinson's disease status (1 for positive, 0 for healthy)
19. **RPDE**: Recurrence period density entropy
20. **DFA**: Detrended fluctuation analysis
21. **spread1**: Measures of vocal fundamental frequency variation
22. **spread2**: Measures of vocal fundamental frequency variation
23. **D2**: Correlation dimension
24. **PPE**: Pitch period entropy

### Usage

1. **Download the Dataset**: Access the above uploaded dataset .

2. **File Format**: The dataset is provided in a CSV format, facilitating seamless integration for training and evaluation.

3. **Data Exploration**: Explore the dataset to understand the distribution of features and labels before utilizing it for model training.

### Model Evaluation

- **Accuracy on Training Data**: 88.46%
- **Accuracy on Test Data**: 87.18%


## How to Contribute

Contributions are welcome! Whether you're enhancing the model, adding features, or improving documentation, follow the standard GitHub workflow. Fork the repository, create a branch, make changes, and submit a pull request.
