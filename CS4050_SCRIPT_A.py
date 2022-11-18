# Script A: Data Cleaning and Merge

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

movies_df = pd.read_csv("data_movie.csv")

############ Data Exploration and Outlier Removal (1.5x IQR) ########

# Get descriptive stats
movies_df.describe()

# Check for duplicates
movies_df[movies_df.duplicated(keep=False)] # No duplicates found

# Check data types
movies_df.dtypes

# Check for missing values
movies_df.isnull().sum() # Key features don't seem to have missing values

# Dropping columns with excessive nans: genre_3, genre_4
movies_df.drop(columns = ["genre_3", "genre_4"], axis = 1, inplace=True)

#Check for outliers using IQR
from pandas.api.types import is_numeric_dtype
for col in movies_df.columns:
    if is_numeric_dtype(movies_df[col]) == True:
        series = movies_df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = 0
        for x in series:
            if x > (q3 + (1.5 * iqr)) or x < (q1 - (1.5*iqr)):
                outliers += 1
                
        print(series.name + " has " + str(outliers) + " outliers")


# Removing outliers based on IQR
from pandas.api.types import is_numeric_dtype
for col in movies_df.columns:
    if is_numeric_dtype(movies_df[col]) == True:
        series = movies_df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        for x in series:
            if x > (q3 + (1.5 * iqr)) or x < (q1 - (1.5*iqr)):
                series.replace(x, np.nan, inplace=True) # Replace outliers with nan
                
movies_df.isnull().sum()

# Drop empty rows
movies_df.dropna(axis = 0,inplace=True)

############ Plotting Visualizations ########

# Histograms
hist_columns = ["mpaa", "genre_1", "genre_2"]

for col in hist_columns:
    movies_df[col].value_counts().sort_values().plot(kind = 'bar', title = f"Histogram of {col}")
    plt.show()

# Descriptive Stats
movies_df.describe()

# Descriptive Graphs
from pandas.api.types import is_numeric_dtype
def visualize_srs(series, name):
    if is_numeric_dtype(series) == True:
        fig, axes = plt.subplots(nrows = 3, ncols =1)
        series.plot(kind = "box", ax = axes[0], figsize = (10,10), title = f"Descriptive plots of {name}")
        series.plot(kind = "kde", ax = axes[1])
        series.plot(kind = "hist", ax = axes[2])
    else:
        return

for col in movies_df.columns:
    visualize_srs(movies_df[col], col)



############ Merge Data Sets on movie_id ########
ratings_df = pd.read_csv("data_rating.csv")

# Check for duplicates
ratings_df[ratings_df.duplicated(keep=False)]

# Check dtypes
ratings_df.dtypes

# Check empty values
ratings_df.isnull().sum()

# Merge on movie_id
merged_df = movies_df.merge(ratings_df, right_on = "movie_id", left_on = "movie_id", how = "right") # Merge based on movie_id
merged_df 

# Check missing values
merged_df.isnull().sum() # Missing values are from ratings for movies that do not exist (movies were removed during the above cleaning step for movie_df)

# Remove missing values
merged_df.dropna(inplace=True) 

merged_df.to_csv("merged_df.csv")



