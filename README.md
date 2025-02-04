# Predicting Device Prices (Laptops and Smartphones)

## Overview

This project aims to predict the prices of laptops and smartphones based on their features using machine learning techniques, specifically the RandomForestRegressor model. The project implements the **CRISP-DM** methodology for data mining and provides an interactive web interface built with Flask to visualize the results.

## Methodology (CRISP-DM)

1. **Business Understanding**: 
   The goal of the project is to predict device prices (laptops and smartphones) based on specific features like RAM, Storage, CPU, and Brand.

2. **Data Understanding**:
   - The project uses two datasets:
     - **laptops.csv**: Contains features of laptops including brand, model, RAM, storage, etc.
     - **smartphones.csv**: Contains features of smartphones including brand, model, RAM, storage, and color.
   - These datasets are analyzed to understand their structure and identify missing values.

3. **Data Preparation**:
   - Missing values are removed.
   - Features are preprocessed by creating dummy variables for categorical data.

4. **Modeling**:
   - A Random Forest Regressor model is used to predict the prices based on the selected features for both laptops and smartphones.
   - The model is trained and evaluated using Mean Absolute Error (MAE) and R² score.

5. **Evaluation**:
   - The model's performance is evaluated based on MAE and R² score for both laptops and smartphones.

6. **Deployment**:
   - A Flask web application is built to display the results of the predictions and show the comparison between actual and predicted prices with graphs.

## Project Structure

- **app.py**: Main Python file containing the Flask application and machine learning model.
- **static/uploads/**: Folder where uploaded files are stored.
- **static/images/**: Folder where generated plots are stored.
- **templates/index.html**: HTML file for rendering the results on the web interface.

## Requirements

- Python 3.x
- Flask
- pandas
- matplotlib
- scikit-learn
- werkzeug

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
