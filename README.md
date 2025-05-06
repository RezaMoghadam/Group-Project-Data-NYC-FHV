NYC FHV Trip Data Analysis – DSC 232 Group Project
Project Overview
This project explores New York City's High Volume For-Hire Vehicle (FHV) data (Uber, Lyft, etc.) from 2019 to 2022 using PySpark for large-scale analysis.

Dataset
Source: NYC Taxi & Limousine Commission

Format: Parquet (~19 GB compressed)

Services: Uber, Lyft, Via, Juno

[Data Dictionary (PDF)](data_dictionary_trip_records_hvfhs.pdf)

Exploratory Data Analysis (EDA)
Loaded and processed using PySpark on the San Diego Supercomputer (SDSC)

Cleaned rows with nulls and invalid values

Explored patterns in trip duration, distance, tips, and driver pay

Visualized demand by time, trip metrics, and correlations

Preprocessing Steps
Dropped rows with nulls in trip_miles, trip_time, driver_pay

Removed trips with 0 distance or 0 fare

Filtered outliers to improve data quality

Selected relevant features for analysis
