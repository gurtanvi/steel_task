import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(seed = 42)

#Reading the dataset
train_data = '/home/gurleenkaur/steel_task/steel_task/task_data.csv'
with open(train_data, mode = 'r') as f:
    data = pd.read_csv(f)

headers = np.array(list(data.columns.values))
feature_names = headers[2:]
class_labels = data.class_label
feature_values = pd.DataFrame(data,columns=feature_names)

#Exploring the dataset
print(data.describe())
print(data.head())

#Exploring the dataset by visualizing the data to see if there is any correlation between the features
sns.heatmap(feature_values.assign(target = class_labels).corr().round(2), cmap = 'Blues', annot = True).set_title('Correlation matrix', fontsize = 14)
plt.show()

#Random Forest consists of many decision trees which are trained through bagging. Subset of data also called boostrap are fed to decision trees. Random forest uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
rf =  RandomForestClassifier(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True, random_state=42)

rf.fit(feature_values.values, class_labels)

print('Out of bag error:', rf.oob_score_)

# function for creating a feature importance dataframe
def imp_features_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def plot_imp_features(imp_features_df, title):
    imp_features_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_features_df, orient = 'h', color = 'green') \
       .set_title(title, fontsize = 20)

imp_features_table = imp_features_df(feature_values.columns, rf.feature_importances_)
print(imp_features_table)

plot_imp_features(imp_features_table, 'Default feature importance (scikit-learn)')
plt.show()

#Other method for checking feature importance : XGBoost Classifier

xgb = XGBClassifier(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True, random_state=42)
xgb.fit(feature_values, class_labels)
# feature importance
print(xgb.feature_importances_)

# plot using default scikit learn feature selection 
plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
plt.show()
