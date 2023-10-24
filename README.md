# Language_Identification

This Python script is a language detection model that can classify text into different languages. It utilizes the scikit-learn library and a logistic regression classifier for this task.

The `NLP_4.py` module is working with `Language Detection.csv` file as the dataset and `NLP_5.py` module is working with `train.csv` file.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data](#data)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Testing](#testing)
  - [Predicting](#predicting)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The code in this repository is designed to detect the language of a given text. It uses a logistic regression classifier trained on a labeled dataset to predict the language of the text. The script can be used to classify text into a variety of languages, making it useful for language identification tasks.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Language_Identification.git


### Installation with Conda

You can install the required Python packages using `conda`. If you don't have conda installed, you can download and install Miniconda or Anaconda from their official websites.

```bash
conda create -n language-detection-env python=3.x
conda activate language-detection-env
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib

## Usage

### Data

The code expects two CSV files: `train.csv` and `test.csv`, which contain the training and testing data, respectively.

### Preprocessing

The script performs text preprocessing, including the removal of symbols, numbers, and English letters to prepare the text data for training.

### Training

The model is trained using a logistic regression classifier and a TF-IDF vectorizer. It learns to classify text into various languages using the training data.

### Testing

The model's accuracy is evaluated using the test data, and a confusion matrix is displayed to assess the classification results.

