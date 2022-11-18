# Script B: Feature Selection/Engineering and Preparation for NB Classifier

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

# Reading Merged Df
train0df = pd.read_csv("merged_df.csv")

# Drop irrelevant features 
train0df.drop(columns = ["trivia", "release", "movie_id"], inplace=True) # dropping index and unused features

############ Feature Selection ########
# Check correlation between numerical features and ratings
train0df.corr()
corr = train0df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

############ Feature Engineering ########
# ## Categorizing numerical features (4 categories)
# Recode the feature (variable) values into quartiles, such that values less than the 25th percentile are assigned a 1, 25th to 50th a 2, 50th to 75th a 3 and greater than the 75th percentile a 4. Thus a single object will deposit one count in bin Q1, Q2, Q3, or Q4.

# Select numeric features 
numeric_features = train0df.select_dtypes(np.number).columns.tolist()
numeric_features.remove("rating") # Remove rating from list of numeric features to convert to categories, since ratings follow it's own rules for categorization
numeric_features

# Categorizing numeric features using quartiles
train1df = train0df.copy()
for col in numeric_features:
    train1df[col] = pd.qcut(x = train1df[col], q = 4, labels = ["1", "2", "3", "4"]) # Cut and relabel based on quartiles



# Converting Ratings to categories
def categorize_ratings(row):
    if row >= 4.0:
        return "TRUE"
    else:
        return "FALSE"

train1df["rating_cat"] = train1df["rating"].apply(categorize_ratings)
train1df.drop(columns = ["rating"], inplace=True)
train1df.head()
train1df["rating_cat"].hist()

############ Changing dtypes ########
# Convert all dtypes to Category
for col in train1df.columns:
    train1df[col] = train1df[col].astype("category")
train1df.dtypes

############ Separating features and labels ########
X = train1df.iloc[:,0:-1] # X is the features in our dataset
y = train1df.iloc[:,-1]   # y is the Labels in our dataset


# Converting features from string to numerical using encoder
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder() 
X_encoded = encoder.fit_transform(X)

X_encoded = pd.DataFrame(X_encoded, columns = X.columns)


############ Performing Train Test Split ########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3) 


############ Initial Model Building ########
# Use all available features
from sklearn.naive_bayes import CategoricalNB
model = CategoricalNB().fit(X_train, y_train) #fitting our model
predicted_y = model.predict(X_test) #now predicting our model to our test dataset
from sklearn.metrics import f1_score
# F measure score
f1_score_all = f1_score(y_test, predicted_y, pos_label='TRUE') 
print (f1_score_all)

