import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
from time import time
from sklearn import metrics
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    df = pd.read_csv('heart_data.csv') # Reading the csv file.

    # Exercise A #

    sub_df = df.loc[:, ['age', 'cp', 'trestbps', 'thalach', 'chol', 'target']] # Sub dataframe to work with.
    target = sub_df.loc[:, 'target']                                           # Target


    # Exercise B #

    ''' This comment section is the first try of Exercise B
    
    sub_df.loc[sub_df['age'].astype(int) > 55, 'age'] = 'older person'
    print(sub_df.head())
    sub_df.loc[sub_df['age'].astype(int) < 56, 'age'] = 'younger person'
    
    '''
    sub_df['age'] = sub_df['age'].astype(str) # Changing the type of age column to string.

    # Changing the age column's data
    for index, row in sub_df.iterrows():
        if (int(sub_df.age[index]) > 55):
            sub_df.at[index, 'age'] = 'older person'
        else:
            sub_df.at[index, 'age'] = 'younger person'

    # Exercise C #

    sub_df = sub_df.apply(LabelEncoder().fit_transform) # Label Encoding

    # Exercise D #

    # Train-Test split part.
    trainData, testData, trainTarget, testTarget = train_test_split(sub_df.drop(['target'], axis=1), target,
                                                                    test_size=0.2, random_state=0)

    # Exercise E #

    # Taking copy of Train-Test to not mess with original data.
    ripperTrainData    = trainData.copy()
    ripperTestData     = testData.copy()
    ripperTrainTarget  = trainTarget.copy()
    ripperTestTarget   = testTarget.copy()

    ripper = lw.RIPPER()                                              # Ripper creation.

    ripperStartTime  = time()                                         # Start time of fit process with Ripper.
    ripper.fit(ripperTrainData, ripperTrainTarget)                    # Ripper's fit process.
    ripperEndTime    = time()                                         # End time of fit process with Ripper.
    ripperScore      = ripper.score(ripperTestData, ripperTestTarget) # Ripper score calculation.


    print("\nElapsed time for ripper algorithm is:", ripperEndTime - ripperStartTime)
    print("Accuracy of Ripper algorithm is:", ripperScore)

    # Exercise F #

    # Taking copy of Train-Test to not mess with original data.
    treeTrainData   = trainData.copy()
    treeTestData    = testData.copy()
    treeTrainTarget = trainTarget.copy()
    treeTestTarget  = testTarget.copy()

    tree = DecisionTreeClassifier()                                     # Decision Tree creation.

    treeStartTime  = time()                                             # Start time of fit process with decision tree.
    tree           = tree.fit(treeTrainData, treeTrainTarget)           # Decision tree's fit process
    treeEndTime    = time()                                             # End time of fit proces with decision tree.
    prediction     = tree.predict(treeTestData)                         # Prediction process with decision tree.
    testAccuracy   = metrics.accuracy_score(treeTestTarget, prediction) # Accuracy score of decision tree.


    print("\nElapsed time for decision tree algorithm is:", treeEndTime - treeStartTime)
    print("Accuracy of decision tree algorithm is: ", testAccuracy)

    # Exercise G #

    # AUC score calculation of ripper and decision tree

    ripperProbs  = ripper.predict_proba(ripperTestData)
    ripperProbs  = ripperProbs[:, 1]
    ripperAUC    = roc_auc_score(ripperTestTarget, ripperProbs)

    print("\nAUC value of ripper algorithm is: ", ripperAUC)

    treeProbs    = tree.predict_proba(treeTestData)
    treeProbs    = treeProbs[:, 1]
    treeAUC      = roc_auc_score(treeTestTarget, treeProbs)

    print("AUC value of tree algorithm is: ", treeAUC)
