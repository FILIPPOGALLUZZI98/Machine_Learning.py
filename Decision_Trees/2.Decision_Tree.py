import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Supponiamo di avere il dataset 'df'
# Splittiamo il dataset 
X_train, X_val, y_train, y_val = train_test_split(df[features], df['target'], train_size = 0.8, random_state = RANDOM_STATE)

# min_samples_split: The minimum number of samples required to split an internal node. Choosing a higher min_samples_split 
# can reduce the number of splits and may help to reduce overfitting
# max_depth: The maximum depth of the tree. Choosing a lower max_depth can reduce the number of splits and may help to reduce overfitting.
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] 
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] 

# Facciamo i seguenti algoritmi per la scelta dei parametri
accuracy_list_train = []; accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
plt.title('Train x Validation metrics'); plt.xlabel('min_samples_split'); plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train); plt.plot(accuracy_list_val);plt.legend(['Train','Validation'])
# Note how increasing the the number of min_samples_split reduces overfitting.

accuracy_list_train = []; accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
plt.title('Train x Validation metrics'); plt.xlabel('max_depth'); plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list); plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val); plt.legend(['Train','Validation'])
# We can see that in general, reducing max_depth can help to reduce overfitting.


# So we can choose the best values for these two hyper-parameters for our model
min = 50; max = 5  ## da valutare in base a quello che si è visto nei casi precedenti
decision_tree_model = DecisionTreeClassifier(min_samples_split = 50, max_depth = 3,
                                             random_state = RANDOM_STATE).fit(X_train,y_train)
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")












