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


## Environmental Setup and Data Download
We use PySpark to process the NYC For-Hire Vehicle (FHV) dataset on SDSC's Expanse platform.  
The environment is configured with increased memory and executor resources to handle the large-scale data efficiently.

This setup includes:
- PySpark 
- Spark session initialization with custom memory and executor settings
- Configuration for parallel processing

```python
import pyspark
print("Using PySpark version:", pyspark.__version__)

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.instances", "5") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()
```

Only the Notebook for Milestone 2 includes the necessary packages and a cell to download the data locally.  
The rest of the notebooks assume that the data is in the local (Data) folder.

Below cell installs the packages inside jupyter notebook 

```python
%pip install pyspark pandas matplotlib seaborn
```

Below cell downloads the files to local Data folder
```python
# Setup & conditional install/download
import os
import sys
import subprocess
from pathlib import Path

# Where the data is expected to be downloaded for this notebook
data_dir = Path.cwd() / "Data"

if not data_dir.exists():
    # Install the Kaggle CLI into this same env
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "kaggle"],
        check=True
    )

    # Create the Data/ directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Tell Kaggle CLI where to look for kaggle.json (in the notebook folder)
    os.environ['KAGGLE_CONFIG_DIR'] = str(Path.cwd()) # Change this if you want to use a different location

    # Verify kaggle.json is in place
    kaggle_json = Path(os.environ['KAGGLE_CONFIG_DIR']) / "kaggle.json"

    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Couldn't find kaggle.json at {kaggle_json}. Download it from your Kaggle account (API section) and place it there."
        )

    # Download & unzip into Data/
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "jeffsinsel/nyc-fhvhv-data",
            "-p", str(data_dir),
            "--unzip"
        ],
        check=True
    )
    print("Kaggle CLI installed, data downloaded into Data/ directory.")
else:
    print("Data directory already exists. Skipping install & download.")
```

The full NYC For-Hire Vehicle (FHV) trip dataset can be accessed on Kaggle:

[NYC TLC Trip Record Data on Kaggle](https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data)


## Contributers

[@RezaMoghadam](https://github.com/RezaMoghadam)
[@avakilov](https://github.com/avakilov)
[@ddkrutkin](https://github.com/ddkrutkin)
[@kpacker77](https://github.com/kpacker77)
