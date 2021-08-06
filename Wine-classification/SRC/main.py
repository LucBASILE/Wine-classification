import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preparing_data_to_complete(path='../DATA/Data_ToComplete.csv'):
    # Get the data from CSV file
    data = pd.read_csv(path, delimiter=';')

    # We don't need wine label in this case
    data.drop('Wine label', axis=1, inplace=True)

    # Check if everything is good/load
    #print(data.head(5))

    X = data.iloc[:, :-1].values

    return X, X


def preparing_data(path='../DATA/Wine_Dataset.csv'):
    # Get the data from CSV file
    data = pd.read_csv(path, delimiter=';')

    # We don't need wine label in this case
    data.drop('Wine label', axis=1, inplace=True)

    # Check if everything is good/load
    #print(data.head(5))

    # Separate Label and Inputs
    X = data.iloc[:, :-1].values
    Y = data['quality']

    return X, Y


def RFT_search_param(X, y, n):
    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv=n, n_jobs = -1, verbose = 2)
    grid_search.fit(X,  y)
    return grid_search.best_params_


def SVC_search_param(X, y, n):
    param = {
        'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
    }
    grid_search = GridSearchCV(svm.SVC(), param_grid=param, scoring='accuracy', cv=n, verbose = 2)
    grid_search.fit(X, y)
    return grid_search.best_params_


if __name__ == '__main__':
    print("===== Step 1: Preparing data =====")
    X, Y = preparing_data()

    print("\n\n===== Step 2: Creating test/train sets =====")
    # Split dataset into train/test sets (ratio: +-20%)
    # Using feature: 'train_test_split'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    print("train inputs:", X_train)
    print("train labels", y_train)

    # Apply normalization to optimize result by avoiding broad range of values. (Because many classifiers calculate the distance between two points by the Euclidean distance)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    print("\n\n===== Step 3: Instantiate classifier =====")

    """ SVC model """

    SVC_model = svm.SVC(C=1.4, gamma='auto', kernel='rbf')
    SVC_model.fit(X_train, y_train)
    SVC_prediction_train = SVC_model.predict(X_train)
    SVC_prediction_test = SVC_model.predict(X_test)

    print("SVC train:", accuracy_score(SVC_prediction_train, y_train))
    print("SVC test:", accuracy_score(SVC_prediction_test, y_test))


    """ RandomForestClassifier """

    # bootstrap=True, max_depth=110, max_features=2, min_samples_leaf=3, min_samples_split=8, n_estimators=100
    # RFT_model = RandomForestClassifier(max_depth=10)
    # RFT_model.fit(X_train, y_train)

    # RFT_prediction_train = RFT_model.predict(X_train)
    # RFT_prediction_test = RFT_model.predict(X_test)

    # print("RFT train:", accuracy_score(RFT_prediction_train, y_train))
    # print("RFT test:", accuracy_score(RFT_prediction_test, y_test))


    """ LinearRegression """

    # LIN_model = linear_model.LinearRegression()
    # LIN_model.fit(X_train, y_train)
    # LIN_prediction_train = LIN_model.predict(X_train)
    # LIN_prediction_test = LIN_model.predict(X_test)
    # print("LIN train:", mean_squared_error(LIN_prediction_train, y_train))
    # print("LIN test:", mean_squared_error(LIN_prediction_test, y_test))

    """ Step 4: predict with "ToComplete" dataset """

    Input, data = preparing_data_to_complete()

    Input = sc.fit_transform(Input)

    SVC_prediction = SVC_model.predict(Input)

    complete_data = np.column_stack((data, SVC_prediction))

    #print(SVC_prediction.shape)
    #print(data.shape)

    """ Step 5: create CSV file with all the result"""

    df = pd.DataFrame(data=complete_data, columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"])

    df.to_csv("../result.csv")






