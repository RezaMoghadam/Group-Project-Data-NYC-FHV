### NYC FHV Trip Data Analysis – DSC 232 Group Project

## Milestone One
## Project Overview
This project explores New York City's High Volume For-Hire Vehicle (FHV) data (Uber, Lyft, etc.) from 2019 to 2022 using PySpark for large-scale analysis.


## Dataset
Source: NYC Taxi & Limousine Commission

Format: Parquet (~19 GB compressed)

Services: Uber, Lyft, Via, Juno

[Data Dictionary (PDF)](data_dictionary_trip_records_hvfhs.pdf)

## Milestone Two 
## Exploratory Data Analysis (EDA)


Loaded and processed using PySpark on the San Diego Supercomputer (SDSC)

Cleaned rows with nulls and invalid values

Explored patterns in trip duration, distance, tips, and driver pay

Visualized demand by time, trip metrics, and correlations


## Preprocessing Steps


Dropped rows with nulls in trip_miles, trip_time, driver_pay

Removed trips with 0 distance or 0 fare

Filtered outliers to improve data quality

Selected relevant features for analysis



## Milestone Three
Linear Regression Model

## Original Full Pipeline Notebook
Preprocessing of data  
Training a LR model  
Analyzing the features and their effects
## Step One Notebook
Using the data after Milestone 2 without further preprocessing
Training of model  
Reviewing the performance  
## Step Two Notebook
Major preprocessing based on observations from Original and Step One  
Training on the data  
Reviewing the improvements  
Further preprocessing  
Training on the data  
Reviewing the improvements  
Conclusion on Milestone 3  


## Data Download

The full NYC For-Hire Vehicle (FHV) trip dataset can be accessed on Kaggle:

[NYC TLC Trip Record Data on Kaggle](https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data)


## Contributers

[@RezaMoghadam](https://github.com/RezaMoghadam)
[@avakilov](https://github.com/avakilov)
[@ddkrutkin](https://github.com/ddkrutkin)
[@kpacker77](https://github.com/kpacker77)
