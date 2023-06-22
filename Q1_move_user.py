import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

movie_types = {"Action":   0,
              "Adventure":1,
              "Animation":2,
              "Children's":3,
              "Comedy":4,
              "Crime":5,
              "Documentary":6,
              "Drama":7,
              "Fantasy":8,
              "Film-Noir":9,
              "Horror":10,
              "Musical":11,
              "Mystery":12,
              "Romance":13,
              "Sci-Fi":14,
              "Thriller":15,
              "War":16,
              "Western":17
              }

def join_l(l, sep):
    li = iter(l)
    string = str(next(li))
    for i in li:
        string += str(sep) + str(i)
    return string

def load_rating_data(root_path, n_users, n_movies):
    """
    Load rating data from file and also return the number of
    ratings for each movie and movie_id index mapping
    @param data_path: path of the rating data file
    @param n_users: number of users
    @param n_movies: number of movies that have ratings
    @return: rating data in the numpy array of [user, movie];
    movie_n_rating, {movie_id: number of ratings};
    movie_id_mapping, {movie_id: column index in
    rating data}
    """
    rates = np.zeros([n_users, n_movies], dtype=np.float32)
    # movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(root_path+"/ratings.dat", 'r') as file:
        for line in file.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1 
            # if movie_id not in movie_id_mapping:
            #     movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            rates[user_id, movie_id] = rating
            if rating > 0.0:
                movie_n_rating[movie_id] += 1
    
    movies = {}
    with open(root_path+"/movies.dat", 'r') as file:
        for line in file.readlines():
            movie_id, title, genres = line.split("::")
            t = genres.strip().split('|')
            g = []
            for name in t:
                g.append(movie_types[name])
            movies[int(movie_id)-1] = [title, join_l(g,'|')]
    
    users = {}
    with open(root_path+"/users.dat", 'r') as file:
        for line in file.readlines():
            user_id, gender, age, occup, _ = line.split("::")
            users[int(user_id)-1] = [1 if gender=='M' else 0, age, occup]

    return rates, movie_n_rating, movies, users 

def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')

def ROC(prediction_prob):
    pos_prob = prediction_prob[:, 1]
    thresholds = np.arange(0.0, 1.1, 0.05)
    true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    for pred, y in zip(pos_prob, Y_test):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                # if truth and prediction are both 1
                if y == 1:
                    true_pos[i] += 1
                # if truth is 0 while prediction is 1
                else:
                    false_pos[i] += 1
            else:
                break
    return true_pos, false_pos

def merged_info(rates, movie_n_rating, movies, users):
    movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
    
    X_raw = np.delete(rates, movie_id_most,axis=1)
    Y_raw = rates[:, movie_id_most]

    tmp = np.zeros((X_raw.shape[0],X_raw.shape[1]+4)) # Add fav, gender, age, occup
    tmp[:,:-4] = X_raw.copy()
    ignores = []
    for uid in range(tmp.shape[0]):
        if uid not in users.keys():
            ignores.append(uid)
            continue
        tmp[uid,-3:] = users[uid]

    if len(ignores) != 0:
        for uid in ignores:
            tmp = np.delete(tmp, uid, axis=0)

    tmp = tmp[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]
    for uid in range(tmp.shape[0]):
        nums = np.zeros(len(movie_types))
        for mid in range(tmp.shape[1]-3):
            if tmp[uid,mid] != 0 and mid in movies.keys():
                _t = movies[mid][1].split('|')
                for i in _t:
                    nums[int(i)] += 1
        tmp[uid,-4] = np.argmax(nums)
    
    X = tmp[:]
    recommend = 3
    Y[Y <= recommend] = 0
    Y[Y > recommend] = 1
    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    print(f'{n_pos} positive samples and {n_neg} negative samples.')

    return X , Y

if __name__ == "__main__":
    data_path = 'ml-1m'
    n_users = 6040
    n_movies = 3952
    rates, movie_n_rating, movies, users = load_rating_data(data_path, n_users, n_movies)
    # print(movies[20])

    display_distribution(rates)

    print(f'#Available movies={len(movies)}, expected={n_movies}\n#Available users={len(users)}, expected={n_users}')

    X, Y = merged_info(rates, movie_n_rating, movies, users)
    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

    display_distribution(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(len(Y_train), len(Y_test))

    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)
    prediction_prob = clf.predict_proba(X_test)
    prediction = clf.predict(X_test)
    print(prediction[:10])
    accuracy = clf.score(X_test, Y_test)
    print(f'The accuracy is: {accuracy*100:.1f}%')

    true_pos, false_pos = ROC(prediction_prob)
    n_pos_test = (Y_test == 1).sum()
    n_neg_test = (Y_test == 0).sum()
    true_pos_rate = [tp / n_pos_test for tp in true_pos]
    false_pos_rate = [fp / n_neg_test for fp in false_pos]
    plt.figure()
    lw = 2
    plt.plot(false_pos_rate, true_pos_rate,color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    pos_prob = prediction_prob[:, 1]
    print(f'AUC={roc_auc_score(Y_test, pos_prob)}')

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