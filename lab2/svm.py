import pandas as pd
from sklearn.svm import SVC


def main():
    # load data
    data = pd.read_csv("svm-data.csv", header=None)

    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]

    # training
    clf = SVC(C=100000, random_state=241, kernel='linear')
    clf.fit(x, y)

    indices = [i + 1 for i in clf.support_]
    print(*indices, sep=',')


if __name__ == "__main__":
    main()
