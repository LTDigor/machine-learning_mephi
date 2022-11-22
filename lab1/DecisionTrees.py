import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def main():
    # load data
    data = pd.read_csv("titanic.csv")

    # clear data from NaN
    data = data.dropna().reset_index(drop=True)

    # select the desired columns
    df = data[['Pclass', 'Fare', 'Age', 'Sex']]

    # replace sex with number
    df = df.replace(to_replace=['male', 'female'], value=[0, 1])

    # get target variable
    survived = data['Survived']

    # training
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(df, survived)

    # get the 2 most important features
    importances = clf.feature_importances_
    features_importances = list(zip(importances, df.columns.values.tolist()))
    features_importances.sort(key=lambda x: x[0], reverse=True)

    # show result
    print(features_importances[0][1], features_importances[1][1], sep=',')


if __name__ == "__main__":
    main()
