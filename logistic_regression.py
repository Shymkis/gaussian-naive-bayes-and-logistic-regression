from matplotlib import pyplot
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.model_selection import KFold
import time
import warnings

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def train_model(train_df, epochs=200, eta=.2, lmbda=0, tol=.00001):
    warnings.filterwarnings("ignore")
    X = train_df.drop("label", axis=1)
    X = np.hstack((np.ones((len(X.index), 1)), X)) # Add intercept and convert to np matrix for efficiency
    Y = train_df["label"]
    n_labels = Y.nunique()
    w = np.zeros((n_labels, X.shape[1]))
    # One vs. all
    for i in range(n_labels):
        for _ in range(epochs):
            diff = (Y == i) - sigmoid(np.dot(X, w[i]))
            step = -eta*lmbda*w[i] + eta*np.dot(diff, X)
            w[i] += step
            if (abs(step) < tol).all():
                break
    return w

def predict(lr_model, test_df):
    X = np.hstack((np.ones((len(test_df.index), 1)), test_df)) # Add intercept and convert to np matrix for efficiency
    return np.argmax(X @ lr_model.T, axis=1)

def prediction_accuracy(labels, predictions):
    return np.mean(labels == predictions) * 100

def main():
    df = pd.read_csv("digits.csv")

    # Cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=517)
    train_accuracy = test_accuracy = time_elapsed = 0
    for train_index, test_index in kf.split(df):
        start = time.time()

        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)

        # Training
        lr_model = train_model(train_df)

        # Display model
        # for i in range(10):
        #     pyplot.subplot(2, 5, 1 + i)
        #     pyplot.imshow(lr_model[i][1:].reshape((28, 28)), cmap=pyplot.get_cmap('gray'))
        #     pyplot.axis("off")
        # pyplot.show()

        # Make predictions on training set and testing set independently
        train_predictions = predict(lr_model, train_df.drop("label", axis=1))
        test_predictions = predict(lr_model, test_df.drop("label", axis=1))

        end = time.time()

        # Calculate accuracies
        train_accuracy += prediction_accuracy(train_df["label"], train_predictions)
        test_accuracy += prediction_accuracy(test_df["label"], test_predictions)
        time_elapsed += end - start
    print("Average train accuracy:", round(train_accuracy / k, 2))
    print("Average test accuracy:", round(test_accuracy / k, 2))
    print("Average time elapsed:", round(time_elapsed / k, 2))

if __name__ == "__main__":
    main()
