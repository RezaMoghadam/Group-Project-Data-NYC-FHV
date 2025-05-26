### NYC FHV Trip Data Analysis – DSC 232 Group Project

## Milestone One
## Project Overview
This project explores New York City's High Volume For-Hire Vehicle (FHV) data (Uber, Lyft, etc.) from 2019 to 2022 using PySpark for large-scale analysis.


## Dataset
Source: NYC Taxi & Limousine Commission

Format: Parquet (~19 GB compressed)

Services: Uber, Lyft, Via, Juno

[Data Dictionary (PDF)](data_dictionary_trip_records_hvfhs.pdf)

## Milestone Two: Exploratory Data Analysis (EDA)
- Loaded and processed using PySpark on the San Diego Supercomputer (SDSC)
- Cleaned rows with nulls and invalid values
- Explored patterns in trip duration, distance, tips, and driver pay
- Visualized demand by time, trip metrics, and correlations

### Preprocessing Steps
- Dropped rows with nulls in trip_miles, trip_time, driver_pay
- Removed trips with 0 distance or 0 fare
- Filtered outliers to improve data quality
-Selected relevant features for analysis

## Milestone Three: Linear Regression Model
### Original Full Pipeline Notebook
- Preprocessing of data  
- Training linear regression model  
- Analyzing the features and their effects

### Step One Notebook
- Using the data after Milestone 2 without further preprocessing
- Training our linear regression model  
- Reviewing the performance  

### Step Two Notebook
- Major preprocessing based on observations from Original and Step One  
- Training on the data  
- Reviewing the improvements  
- Further preprocessing  
- Training on the data  
- Reviewing the improvements  
- Conclusion for Milestone 3  

## Environmental Setup and Data Download
We use PySpark to process the NYC For-Hire Vehicle (FHV) dataset on SDSC's Expanse platform.  

The environment is configured with increased memory and executor resources to handle the large-scale data efficiently.

This setup includes:
- PySpark 
- Spark session initialization with custom memory and executor settings
- Configuration for parallel processing

The Notebook for Milestone 2 (in the Milestone-2 branch) includes the necessary packages and a cell to download the data locally.  
The rest of the notebooks assume that the data is in the local (Data) folder.

We include the dependencies cell (if a user is planning to run our notebooks outside of SDSC) and data download; however, on SDSC, we already had the "Data" folder and all downloaded parquet files. 

The cell below installs the packages inside jupyter notebook (if a Kernal does not already have them installed):

```python
%pip install pyspark pandas matplotlib seaborn
```

The cell below contains an optional install for users who have not downloaded the data manually into a folder called "Data". If a folder called "Data" exists within the same directory as the notebooks, we assume the parquet files exist within that directory. 
```py
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

Finally, we have our Spark Session builder command below:

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

The full NYC For-Hire Vehicle (FHV) trip dataset can be accessed on Kaggle:

[NYC TLC Trip Record Data on Kaggle](https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data)


## Contributers

[@RezaMoghadam](https://github.com/RezaMoghadam)
[@avakilov](https://github.com/avakilov)
[@ddkrutkin](https://github.com/ddkrutkin)
[@kpacker77](https://github.com/kpacker77)
