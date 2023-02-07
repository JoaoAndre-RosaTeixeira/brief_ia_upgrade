from scipy.stats import uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import scipy.io as sio
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from features_functions import compute_features

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Set the paths to the files
data_path = "Data/"

# Names of the classes
classes_paths = ["Cars/", "Trucks/"]
classes_names = ["car", "truck"]
cars_list = [4,5,7,9,10,15,20,21,23,26,30,38,39,44,46,48,51,52,53,57]
trucks_list = [2,4,10,11,13,20,22,25,27,30,31,32,33,35,36,39,40,45,47,48]
nbr_of_sigs = 20 # Nbr of sigs in each class
seq_length = 0.2 # Nbr of second of signal for one sequence
nbr_of_obs = int(nbr_of_sigs*10/seq_length) # Each signal is 10 s long

# Go to search for the files
learning_labels = []
for i in range(2*nbr_of_sigs):
    if i < nbr_of_sigs:
        name = f"{classes_names[0]}{cars_list[i]}.wav"
        class_path = classes_paths[0]
    else:
        name = f"{classes_names[1]}{trucks_list[i - nbr_of_sigs]}.wav"
        class_path = classes_paths[1]

    # Read the data and scale them between -1 and 1
    fs, data = sio.wavfile.read(data_path + class_path + name)
    data = data.astype(float)
    data = data/32768

    # Cut the data into sequences (we take off the last bits)
    data_length = data.shape[0]
    nbr_blocks = int((data_length/fs)/seq_length)
    seqs = data[:int(nbr_blocks*seq_length*fs)].reshape((nbr_blocks, int(seq_length*fs)))

    for k_seq, seq in enumerate(seqs):
        # Compute the signal in three domains
        sig_sq = seq**2
        sig_t = seq / np.sqrt(sig_sq.sum())
        sig_f = np.absolute(np.fft.fft(sig_t))
        sig_c = np.absolute(np.fft.fft(sig_f))

        # Compute the features and store them
        features_list = []
        N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], fs)
        features_vector = np.array(features_list)[np.newaxis,:]

        if k_seq == 0 and i == 0:
            learning_features = features_vector
            learning_labels.append(classes_names[0])
        elif i < nbr_of_sigs:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[0])
        else:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[1])

print(learning_features.shape)
print(len(learning_labels))


def run_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



    models = [
        DecisionTreeClassifier(random_state=1),
        RandomForestClassifier(random_state=1),
        SVC(probability=False),
        KNeighborsClassifier(),
        BaggingClassifier(random_state=1),
        LogisticRegression(random_state=1),
        MLPClassifier(random_state=1),
        GradientBoostingClassifier(random_state=1),
        GaussianNB(),
        AdaBoostClassifier(random_state=1)
    ]

    model_names = [
        'Decision Tree',
        'Random Forest',
        'SVM',
        'KNN',
        'BaggingClassifier',
        'LogisticRegression',
        'MLPClassifier',
        'GradientBoostingClassifier',
        'GaussianNB',
        'AdaBoostClassifier'
    ]

    param_grids = [
        {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        {
            'n_estimators': [50, 100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        {
            'C': [0.1, 1, 10, 35, 100],
            'kernel': ['linear', 'rbf'],
            'class_weight': [None, 'balanced']
        },
        {
            'n_neighbors': [5, 10, 20],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        {
            'base_estimator': [DecisionTreeClassifier()],
            'n_estimators': [50, 100, 150],
            'max_samples': [0.5, 0.7, 1.0],
            'max_features': [0.5, 0.7, 1.0],
            'bootstrap_features': [True, False]
        },
        {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': uniform(loc=0, scale=4),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [50, 100, 250, 500],
            'fit_intercept': [True, False]
        },
        {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.05],
            'batch_size': ['auto', 32, 64, 128],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        },
        {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.1, 0.01, 0.05],
            'max_depth': [1, 3, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.5, 0.7, 1.0],
        },
        {
            'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05],
        },
        {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.1, 0.01, 0.05],
            'algorithm': ['SAMME', 'SAMME.R'],
        }

    ]

    results = []

    for model, model_name, param_grid in zip(models, model_names, param_grids):
        randomized_search = RandomizedSearchCV(model, param_grid, cv=5, n_jobs=-1)
        randomized_search.fit(X_train, y_train)
        best_model = randomized_search.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((model_name, accuracy))

    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    results_df = results_df.sort_values('Accuracy', ascending=False)

    return results_df

# Example usage
df_models = run_models(learning_features, learning_labels)

df_models.to_csv("models_accuracy.csv")

