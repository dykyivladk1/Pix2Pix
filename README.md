# Pix2Pix Implementation

This repository is my personal implementation of the Pix2Pix Generative Adversarial Network (GAN). The Pix2Pix GAN is specifically designed for image-to-image translation tasks, leveraging a conditional generative adversarial network to transform an input image into a desired output image.

## Getting Started

Follow these steps to use this implementation:

### Prerequisites

Ensure you have Python installed on your system. This code is compatible with Python 3.6 and newer versions.

### Dataset

For training and testing the Pix2Pix model, you'll need a dataset. Download the Pix2Pix dataset from Kaggle using the following link:

[Pix2Pix Dataset on Kaggle](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset)

After downloading, place the dataset in an appropriate directory within your project structure, such as `./data`.

### Installation

1. **Clone the repository** to your local computer:

    ```
    git clone https://github.com/dykyivladk1/Pix2Pix.git
    ```


2. **Install the required dependencies**. It's recommended to create and use a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Running the Flask Application**

    To run the Flask application, execute the following command from the root directory of the project:

    ```
    python flask_app/app.py
    ```
4. **Training model**
   To train a model for custom dataset, you can use the following command:
   ```
   python scripts/train.py --train_dir <train_path> --val_dir <val_path>
   ```
