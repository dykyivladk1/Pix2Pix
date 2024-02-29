# Pix2Pix Implementation

This repository contains my personal implementation of the Pix2Pix Generative Adversarial Network (GAN). The Pix2Pix GAN is designed for image-to-image translation tasks, leveraging a conditional generative adversarial network to convert an input image to a desired output image.

## Getting Started

To use this implementation, follow the steps below:

### Prerequisites

Ensure you have Python installed on your system. This code was tested with Python 3.6 and above.

### Dataset

You need to download the dataset for training and testing the Pix2Pix model. Use the following link to download the dataset from Kaggle:

[Pix2Pix Dataset on Kaggle](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset)

After downloading, make sure to place the dataset in an appropriate directory within your project structure (e.g., `./data`).

### Installation

1. **Clone the repository** to your local machine by using the command below. Make sure to replace `<https://github.com/dykyivladk1/Pix2Pix.git>` with the actual URL of your repository:

    ```
    git clone https://github.com/dykyivladk1/Pix2Pix.git
    ```

2. **Navigate to the repository's directory** on your local machine:

    ```
    cd scripts
    ```

3. **Install the required dependencies**. It's recommended to use a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4. **Running the Flask Application**

To run the Flask application, execute the following command from the root directory of the project:

    ```
    python flask_app/app.py
    ```

