import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt


# Import the 3 dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

np.random.seed(seed = 42)

#df = pd.read_csv("sensor_task/task_data.csv",header=0, index_col=0, quotechar='"',sep='[,;]', engine='python', na_values = ['na', '-', '.', '', ';'] )
train = '/home/gurleenkaur/task_sensor/sensor_task/task_data.csv'
with open(train, mode = 'r') as f:
    data = pd.read_csv(f)

headers = np.array(list(data.columns.values))
names = headers[2:]
y = data.class_label
X = pd.DataFrame(data,columns=names)

print(headers)
print(names)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

rf =  RandomForestClassifier(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True, random_state=42)

rf.fit(X, y)

print('Out of bag error:', rf.oob_score_)

def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'green') \
       .set_title(title, fontsize = 20)



base_imp = imp_df(X.columns, rf.feature_importances_)
print(base_imp)

var_imp_plot(base_imp, 'Default feature importance (scikit-learn)')

