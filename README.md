# BreakHis Fine-Tuning with Transformers

## Introduction

This repository focuses on the fine-tuning of models on the BreakHis dataset utilizing the Transformers library from Hugging Face. The BreakHis dataset is an essential resource in the medical imaging domain, particularly for breast cancer histopathological image analysis. Our goal is to harness the power of the Transformers library to enhance the accuracy and efficiency of image processing, thus contributing to advancements in medical diagnostics.

## Setup

### Prerequisites

Before beginning the setup, ensure you have Conda installed on your system. If Conda is not already installed, you can download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

### Dataset

The BreakHis dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis).

### Setting up the Environment

1. **Clone the Repository**

   Clone this repository to your local machine using the following Git command:

   ```bash
   git clone https://github.com/mikkac/breakhis_vit.git
   cd breakhis_vit
   ```

2. **Create a Conda Environment**

   Create a new Conda environment with Python 3.11:

   ```bash
   conda create -n breakhis_env python=3.11
   conda activate breakhis_env
   ```

3. **Install Dependencies**

   Install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that you have a `requirements.txt` file in the root of the project directory containing all the necessary packages.

4. **Verify Installation**

   Verify the installation by checking the installed packages:

   ```bash
   pip list
   ```