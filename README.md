### NYC FHV Trip Data Analysis – DSC 232 Group Project
## Project Overview
This project explores New York City's High Volume For-Hire Vehicle (FHV) data (Uber, Lyft, etc.) from 2019 to 2022 using PySpark for large-scale analysis.


## Dataset
Source: NYC Taxi & Limousine Commission

Format: Parquet (~19 GB compressed)

Services: Uber, Lyft, Via, Juno

[Data Dictionary (PDF)](data_dictionary_trip_records_hvfhs.pdf)

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


## Data Download

The full NYC For-Hire Vehicle (FHV) trip dataset can be accessed on Kaggle:

[NYC TLC Trip Record Data on Kaggle](https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data)

## Regrade Request Update
After the announcement on Canvas regarding re-grade request, we updated our Milestone 2 requirements in the following ways:
- Created a dedicated branch for Milestone 2
- Updated our Jupyter notebook to pip install critical packages (if someone were to run the notebook outside of SDSC and doesn't have the required packages)
- Updated our Jupyter notebook to have an optional lightweight data install cell. We give the users two options to obtain the source data:
    1. Manually creating a "Data" directory and moving the parquet files into that directory
    2. Running our in-notebook cell which will create the directory and download the source data via Kaggle API call. The notebook assumes that if the "Data" directory already exists, then the user has obtained the data manually. 


## Contributers

[@RezaMoghadam](https://github.com/RezaMoghadam)
[@avakilov](https://github.com/avakilov)
[@ddkrutkin](https://github.com/ddkrutkin)
[@kpacker77](https://github.com/kpacker77)
