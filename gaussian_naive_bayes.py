import math
from matplotlib import pyplot
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold
import time

def train_model(train_df):
    label_probs = train_df.label.value_counts()/len(train_df.index)
    feature_label_means = train_df.groupby("label").mean()
    feature_label_stdvs = train_df.groupby("label").std()
    return label_probs, feature_label_means, feature_label_stdvs

def predict(gnb_model, test_df):
    label_probs, feature_label_means, feature_label_stdvs = gnb_model
    label_values = {}
    for label in label_probs.index:
        probs = []
        for feature in test_df.columns:
            x = test_df[feature]
            mean = feature_label_means.at[label, feature]
            stdv = feature_label_stdvs.at[label, feature]
            if stdv == 0:
                continue
            prob = norm.logpdf(x, mean, stdv)
            probs.append(prob)
        label_values[label] =  math.log(label_probs[label]) + np.sum(probs, axis=0)
    return pd.DataFrame(label_values).idxmax(axis=1)
 
def prediction_accuracy(labels, predictions):
    correct = labels.eq(predictions).sum()
    return correct / len(labels.index) * 100

def main():
    # https://www.kaggle.com/c/digit-recognizer
    df = pd.read_csv('digits.csv')

    # Cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=517)
    train_accuracy = test_accuracy = time_elapsed = 0
    for train_index, test_index in kf.split(df):
        start = time.time()

        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)

        # Training
        gnb_model = train_model(train_df)

        # Display model
        # for i in range(10):
        #     pyplot.subplot(2, 5, 1 + i)
        #     pyplot.imshow(gnb_model[1].iloc[i].to_numpy().reshape((28, 28)), cmap=pyplot.get_cmap('gray'))
        # pyplot.show()

        # Make predictions on training set and testing set independently
        train_predictions = predict(gnb_model, train_df.drop("label", axis=1))
        test_predictions = predict(gnb_model, test_df.drop("label", axis=1))

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
