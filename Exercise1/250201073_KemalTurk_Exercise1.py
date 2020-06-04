import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import numpy as np
import scipy.stats


def fitPredictAccuracy(inputCriterion, dataTrain, depositTrain, dataTest, depositTest, depthLimit):
    clf = DecisionTreeClassifier(criterion=inputCriterion, max_depth=depthLimit)
    clf = clf.fit(dataTrain, depositTrain)
    testPredictions = clf.predict(dataTest)
    trainPredictions = clf.predict(dataTrain)
    testAccuracy = metrics.accuracy_score(depositTest, testPredictions)
    trainAccuracy = metrics.accuracy_score(depositTrain, trainPredictions)
    print("Accuracy of test predictions is: ", testAccuracy)
    print("Accuracy for train predictions is: ", trainAccuracy)
    data_accuracy_values.append(testAccuracy)

    return clf

def saveTreeAsPNG(clf, columns, saveName):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(saveName)
    Image(graph.create_png())

def confidanceInterval(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

if __name__ == '__main__':

    df = pd.read_csv('bank_customer.csv')

    data_accuracy_values = []

    # Exercise A #

    df = df.replace(to_replace = ["management", "admin."], value = "white-collar")
    df = df.replace(to_replace = ["services", "housemaid"], value = "pink-collar")
    df = df.replace(to_replace = ["retired", "student", "unemployed", "unknown"], value = "other")
    print("All jobs with counts:\n")
    print(df['job'].value_counts()) # Printing all jobs with counts

    df = df.replace(to_replace = ["other"], value = "unknown")
    print("\nAll poutcomes with counts\n")
    print (df.groupby('poutcome')['poutcome'].count())

    # Exercise B #

    df = df.apply(LabelEncoder().fit_transform)

    # Exercise C and D #

    deposit = df.deposit

    data_1 = df[['age', 'job', 'marital', 'education', 'balance', 'housing', 'duration', 'poutcome']]
    data_1_cols = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'duration', 'poutcome']
    data_1Train, data_1Test, depositTrain1, depositTest1 = train_test_split(data_1, deposit, test_size=0.3, random_state=0)
    #print(data_1.head())

    data_2 = df[['job', 'marital', 'education', 'housing']]
    data_2_cols = ['job', 'marital', 'education', 'housing']
    data_2Train, data_2Test, depositTrain2, depositTest2 = train_test_split(data_2, deposit, test_size=0.3, random_state=0)
    #print(data_2.head())

    # Exercise E #

    print("\n----- Data 1 Entropy -----")
    data1Entropy = fitPredictAccuracy('entropy', data_1Train, depositTrain1, data_1Test, depositTest1, None)

    print("\n----- Data 2 Entropy -----")
    data2Entropy = fitPredictAccuracy('entropy', data_2Train, depositTrain2, data_2Test, depositTest2, None)

    # Exercise F #

    print("\n----- Data 1 Gini -----")
    data1Gini = fitPredictAccuracy('gini', data_1Train, depositTrain1, data_1Test, depositTest1, None)

    print("\n----- Data 2 Gini -----")
    data2Gini = fitPredictAccuracy('gini', data_2Train, depositTrain2, data_2Test, depositTest2, None)

    # Exercise G #

    i = 1

    while(i < 32):

        print("\n+++++ Current max depth value is:", i, "+++++")

        print("\n----- Data 1 Entropy with max depth value of ", i, "-----")
        fitPredictAccuracy('entropy', data_1Train, depositTrain1, data_1Test, depositTest1, i)
        print("\n----- Data 2 Entropy with max depth value of ", i, "-----")
        fitPredictAccuracy('entropy', data_2Train, depositTrain2, data_2Test, depositTest2, i)

        print("\n----- Data 1 Gini with max depth value of ", i, "-----")
        fitPredictAccuracy('gini', data_1Train, depositTrain1, data_1Test, depositTest1, i)
        print("\n----- Data 2 Gini with max depth value of ", i, "-----")
        fitPredictAccuracy('gini', data_2Train, depositTrain2, data_2Test, depositTest2, i)

        i = i + 5

    # Exercise H #

    m, lower, upper = confidanceInterval(data_accuracy_values, 0.95)
    print("\nMean:", m, "\nUpper p:", upper, "\nLower p:", lower)

    # Exercise I #

    saveTreeAsPNG(data1Entropy, data_1_cols, 'data1Entropy.png')
    saveTreeAsPNG(data2Entropy, data_2_cols, 'data2Entropy.png')
    saveTreeAsPNG(data1Gini, data_1_cols, 'data1Gini.png')
    saveTreeAsPNG(data2Gini, data_2_cols, 'data2Gini.png')

