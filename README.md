# Breast Cancer Predictor Using Machine Learning 

Imagine a cutting-edge Breast Cancer Diagnosis app, crafted with powerful machine learning capabilities, tailored to support medical professionals in accurately diagnosing breast cancer. This innovative tool analyzes a comprehensive set of measurements to predict whether a breast mass is benign or malignant, transforming complex data into a clear, visual radar chart. It not only delivers a precise diagnosis but also presents the probability of the mass being benign or malignant, empowering healthcare providers with crucial insights.

Accessible and versatile, the app offers seamless integration with cytology labs, enabling automated data retrieval directly from lab machines for swift analysis. Please note, while the app seamlessly interfaces with lab equipment, the connection to the laboratory machine itself is managed independently. This ensures efficiency and accuracy in diagnosing breast cancer, revolutionizing medical diagnostics with advanced technology at its core.
## Tech Stack
`numpy`
`pandas`
`pickle`
`plotly`
`scikit_learn`
`streamlit`
`altair`
`streamlit`
## Installation

You can run this inside a virtual environment to make it easier to manage dependencies
`conda` to create a new environment and install the required packages

```bash
conda create -n breast-cancer-diagnosis python=3.10 
```
Then, activate the environment:
```bash
conda activate breast-cancer-diagnosis
```
To install the required packages, run:

```bash
pip install -r requirements.txt
```
This will install all the necessary dependencies, including Streamlit

Datasets - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
## Usage

```bash
streamlit run app/main.py
```
