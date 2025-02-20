# Time-Series-Analysis-and-Forecasting-with-Dash-Web-App
This repository captures a project to carry out time series analysis and forecasting and this deployed as a Plotly Dash App. The dataset to be analysed and model is crude oil prices - both West Texas Index (WTI) and Brent. 

**Problem Statement:** 
A time series analysis and forecasting app to be developed to assist an oil and gas producing company make data-driven investment decision for the drilling and subsequent sales of crude oil. 

> Given historical crude oil prices, can we forecast into different future horizons what the price might be? 

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Technology Stack](#tech-stack)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Model Evaluation](#evaluating-multiple-models)
7. [Deployment](#deployment)
    - [Dash App](#Dash-app)
    - [Web Deployment](#azure-deployment)
8.  [Update Workflows](#update-workflows)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)



## Project Overview
The time series analysis and forecasting app to be developed is to assist an oil and gas producing company make data-driven investment decision for the drilling and subsequent sales of crude oil. As the return on investment (ROI) are driven by crude oil prices, an ongoing review of price trends and ability to forecast its direction would be highly beneficial to such organisation. This should decide if and when to drill and possible periods that the best price can be gotten from the crude oil sales.

The following are objectives of the project:

* Create a Extraction, Transformation and Loading (ETL) pipeline for data sourced using the Alpha Vantage API.
* Carry out a time series exploratory data analysis to uncover the characteristics of the dataset
* Based on th EDA experiment with relevant Time Series Forecasting models and tune hyperparameters accordingly for best results.
* Deploy the forecasting model into a production environment. 
* Utilised a modular programming structure using OOP
* Build the dashboard as a Plotly Dash App 
* Deployed the Dash App in a custom domain Flask Web App

  
## Project Structure
```

├── assets
├── artifacts
|   ├── data_ingestion
|   |   ├──WTI
│   |   ├──Brent
│   ├── data_transformation
│   │   ├──WTI 
│   │   ├──Brent 
│   ├── best_model.pkl
├── logs
├── notebooks
|   ├──ETL_EDA_Modelling_Experiment.ipynb
├── src
|   ├── __init__.py
│   ├── components
│   │   ├── __init__.py
│   │   ├── data_ETL.py
│   │   ├── data_transformation.py
│   │   ├── model_training.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── config_entity.py
│   │   ├── config.yaml
│   │   ├── configuration.py
│   │   ├── params.yaml
│   ├── constants
│   │   ├── __init__.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── stage_01_data_ETL.py
│   │   ├── stage_02_data_transformation.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_prediction_pipeline.py
│   │   ├── exception.py
│   │   ├── forms.py
│   │   ├── logger.py
│   ├── notebooks
│   ├── static
│   │   ├──vendor
│   │   |   ├──bootstrap
│   │   |   ├──aos
│   │   |   ├──glightbox
│   │   ├──main.css    
├──utils
│   │   ├── __init__.py
│   │   ├── common.py
├── tests
│   ├── function_class_test.py
├── venv
├── __init__.py
├── .env
├── .gitignore
├── app.py
├── main.py
├──  README.md
├── requirements.txt


```
## Tech Stack
### Dependencies
```
- dash
- plotly
- pandas 
- numpy
- alpha_vantage
- dash_bootstrap_components
- jupyter
- python-dotenv
- seaborn
- matplotlib
- statsmodels
- scikit-learn
- arch
- prophet
- python-box
- dill
- Github (action)

```
### Step-by-Step Implementation
 **Install the required packages:**
    ```
    pip install -r requirements.txt
    ```

## Data Preparation
### Data Sources
- The original data if from the Cleveland data from the UCI machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+disease

- There is also a version on Kaggle. https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset 

### Data Cleaning, Preprocessing and Transformation
The following steps were carried out of this dataset to clean and transform it for use:
- Missing Values:
    * There is no missing values in the dataset
- There was no requirement for transformation as well as there dataset were already encoded for its categorical features
        
### Time Series Exploration Data Analysis 


				

## Model Training
### Model Selection
The following models were selected after using recommendations from researching classification models and subsequent experimenting:

- Logistic regression
- Support Vector Machine Classifier (SVC)
- Random Forest Classifier
- XGBoost
- LightGBM

### Training the Model
- Loading the Configuration
First, load the  model training configuration. This configuration includes settings which are the paths to the training, validation, and test data, as well as the machine learning models and their parameters.
- Loading the Data
Next, load the datasets:
    * Training Data: Used to train the models.
    * Validation Data: Used to tune and validate the models during the training process.
    * Test Data: Used to evaluate the final model's performance.
- Splitting the Data
Split each dataset into:
    * Input Features (X): The data that our model will learn from.
    * Target Labels (y): The actual outcomes we want to predict (whether the patient has heart diesease or not).

### Evaluating Multiple Models
Evaluated several machine learning models to see which one performs the best. 

Each model is tested using the training and validation data. Used a helper function based on ‘GridSearchCV’ called evaluate_models using hyperparameters values set in the `params.yaml` file to check how well each model performs and store the results. 
- Selecting the Best Model
Looked at the performance scores of all the models and choose the one with the highest score. If no model achieves a score of at least 0.6 (out of 1.0), the application raises an error indicating that no suitable model was found.
- Saving the Best Model
Once the best model is identified, it is saved to a file to be used later to make predictions.

This process will be updated in the next version to incorporate a complete MLOps pipeline with automated model training and model Registry 

## Deployment
### Dash App in Flask Web App
- The Flask web framework was used to share the web application for use. A simple web app with a form to collect data points and return the prediction was developed. CSS was used to perform minimal customasation of the Bootstrap template used for the web page. This web app has the capability to email the prediction results to the email address provided and also store the prediction data and result in a database.

Use the link below to access the web app deployed to custom domain- `www.soyinkasowoolu.com`:

https://hdpredictor.azurewebsites.net/

## Update Workflow

1. Update the config_entity in config
2. Update config.yaml in config
3. Update params.yaml in config
4. Update the configuration manager in config
5. Update the components scripts
6. Update the pipeline scripts
7. Update the main files (forms.py, views.py, errors.py, exception.py)
8. Update db_models.py and email.py
8. Update the application.py

## Usage
### Accessing the Deployed Model
- Upon launching the application, the default page is the `About` page.
- Use the `Predict`tab to access the prediction page and provide values for all the data endpoints for the patient using the input boxes, dropdowns etc.
- Click the submit button for prediction by the model. The result is displayed in the web page and also emailed to the email address provided.
  
## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License 

## Acknowledgments
The project code, structure, best practices etc. were inspired and learnt from a myriad of open sources such as :
- Coursework from Zero to Mastery Data Science ML Bootcamp 
- Youtube videos of Kris Naik such as the End to End Machine Learning Project Video 
- Books and literature such as Flask Web Development, Data Science from Scratch etc.
- Udemy trainings
- Github repositories and resources
- Medium.com articles etc.
- Lastly and not the least -ChatGPT!

## Future Work/Advancements

1. Update with an automated training pipeline with model registry using tools such as MLFlow, DVC, CometML.

