# Script C: Model Tuning and Model Evaluation with question_movie.csv
# DISCLAIMER: THE BELOW CODE TAKES AWHILE TO RUN. 
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

######## Check Feature Importance with Permutation Importance #######
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

######## Iteratively testing feature sets #######
top_features = ["user_id", "domestic", "year", "worldwide", "budget", "mpaa", "genre_2", "genre_1", "run_time", "composer", "main_actor_1", "producer", "main_actor_3", "title", "main_actor_2", "main_actor_4", "writer", "director"]

for i in range(1,18):
    temp_feats = top_features[0:i]
    X_train_temp = X_train[temp_feats]
    X_test_temp = X_test[temp_feats]
    model_temp = CategoricalNB().fit(X_train_temp, y_train) #fitting our model
    pred_y_temp = model_temp.predict(X_test_temp) #now predicting our model to our test dataset
    f1_score_temp = f1_score(y_test, pred_y_temp, pos_label='TRUE') 
    print(f"{str(i)} features yields an F1 score of {str(f1_score_temp)}")   

# Confusion Matrix of Best Performing Model
best_features = top_features[0:14]
final_model = CategoricalNB().fit(X_train[best_features], y_train) #fitting our model
final_y_pred = final_model.predict(X_test[best_features]) # now predicting our model to our test dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, final_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
disp.plot()

############ Using Model for question_movie.csv ########
# Results from question_movie.csv
eval_df = pd.read_csv("question_movie.csv")

# Define custom function
def find_movie(row):
    try:
        title = movies_df[movies_df["movie_id"] == row["movie_id"]]["title"].iloc[0] # If movie cannot be found in cleaned movie dataframe, output "FALSE"
    except:
        return "FALSE"
    
    movie_row = train1df[train1df["title"] == title] # Locate from encoded df based on movie_row

    movie_row.drop(columns = ["rating_cat"], inplace=True) # Drop label
    movie_row["user_id"] = row["user_id"] # Include user_id as a feature 
    encoded = encoder.transform(movie_row) # Reuse same encoder from before 
    encode_df = pd.DataFrame(encoded, columns = movie_row.columns)
    return final_model.predict(encode_df[best_features].head(1))[0] # Predict using the current row's data

    
eval_df["recommend"] = eval_df.apply(find_movie, axis = 1) # Apply function to all rows
eval_df

# Plot histogram
eval_df["recommend"].value_counts().plot(kind = "bar")

# Export to CSV
eval_df.to_csv("Evaluation_results.csv")
