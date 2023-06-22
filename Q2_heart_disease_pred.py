import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
# import pandas as pd

def load_heart_data(data_path):
    """
    Load Heart Disease data from file
    @param data_path: path of the rating data file
    @return: valsfeaturs in the numpy array;
    labels in the numpy array (0: absence, 1: presence [values 1,2,3,4])
    """
    labels = []
    featurs = [] 
    with open(data_path, 'r') as file:
        for i,line in enumerate(file.readlines()):
            values = line.split(",")
            if i in [87,166,192,266,287,302]:  # Indexes of missin (?) values
                values[-2 if i in [87,266] else -3] = '0.0' 
            lbl = int(values[-1])
            labels.append(1 if lbl>0 else 0)
            featurs.append(values[:-1])
    return np.array(np.float32(featurs)), np.array(labels)

def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of {"presence" if value else "absence"}: {count}')

if __name__ == "__main__":
    data_path = 'heart/cleveland.data'
    
    #load the disease data from cleveland.data
    X, Y= load_heart_data(data_path)

    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

    display_distribution(Y)

    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    print(f'{n_pos} presence samples and {n_neg} absence samples.')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f'#of train and test samples: {len(Y_train)}, {len(Y_test)}')

    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)
    prediction_prob = clf.predict_proba(X_test)
    print(prediction_prob[0])
    prediction = clf.predict(X_test)
    print(prediction[:10])
    accuracy = clf.score(X_test, Y_test)
    print(f'The accuracy is: {accuracy*100:.1f}%')

    #Tuning the model with cross-validation
    k = 5
    k_fold = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    smoothing_factor_option = [1, 2, 3, 4, 5, 6]
    fit_prior_option = [True, False]
    auc_record = {}
    for train_indices, test_indices in k_fold.split(X, Y):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        for alpha in smoothing_factor_option:
            if alpha not in auc_record:
                auc_record[alpha] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(X_train, Y_train)
                prediction_prob = clf.predict_proba(X_test)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(Y_test, pos_prob)
                auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)
    
    bestModel = [1,True]
    bestAUC = 0.0
    print(f'{"Smoothing":^11}{"fit_prior":^11}{"AUC":^11}')
    for smoothing, smoothing_record in auc_record.items():
        for fit_prior, auc in smoothing_record.items():
            print(f'{smoothing:^11d}{"True" if fit_prior else "False":^11}{auc/k:^.5f}')
            if auc>bestAUC:
                bestModel[0], bestModel[1] = smoothing, True if fit_prior else False
                bestAUC = auc

    clf = MultinomialNB(alpha=bestModel[0], fit_prior=bestModel[1])
    clf.fit(X_train, Y_train)
    pos_prob = clf.predict_proba(X_test)[:, 1]
    print(f'AUC with the best model [{bestModel[0],"True" if bestModel[1] else "False"}]:', roc_auc_score(Y_test,pos_prob))