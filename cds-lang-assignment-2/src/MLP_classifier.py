import os
import sys

from sklearn import metrics
from sklearn.neural_network import MLPClassifier

# Load the stored data 
X_train, X_test, y_train, y_test, X_train_features, X_test_features, feature_names = pd.read_pickle('../out/features.pkl')

# Fit MLP classifier to training set, save it in 'models' folder, 
# and test it on test set. 
classifier_MLP = MLPClassifier(activation = "logistic",
                               hidden_layer_sizes = (20,), 
                               max_iter=1000, 
                               random_state=42).fit(X_train_features, y_train)
dump(classifier_MLP, "../models/classifier_MLP.joblib")

y_pred_MLP = classifier_MLP.predict(X_test_features)


# Calculate evalutation metrics and save them as txt file in 'out'folder 
classifier_MLP_metrics = metrics.classification_report(y_test, y_pred_MLP)
save_metrics(classifier_MLP_metrics)