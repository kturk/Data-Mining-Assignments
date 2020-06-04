import numpy as np
from math import log
import pandas as pd
from sklearn.preprocessing import LabelEncoder

'''
Takes the dataframe and the criterion as arguments and returns the desired calculation result.
'''
def entropy_gini_calc(dataframe, criterion):

    base                   = 2
    len_of_values          = len(dataframe['target'])
    value_list, count_list = np.unique(dataframe['target'], return_counts=True)  # To have value and the counts of them.
    prob_list              = count_list / len_of_values  # To have probabilities of the values.
    result                 = 0.

    # Entropy and Gini calculations.
    if criterion == 'entropy':
        for prob in prob_list:
            result -= prob * log(prob, base)

    elif criterion == 'gini':
        for prob in prob_list:
            result += prob * prob
        result = 1 - result

    else:
        print("Incorrect Criterion!")

    return result

'''
Takes the dataframe, criterion, and the specific attribute as arguments and returns the desired calculation result.
'''
def split_calc(dataframe, column, criterion):

    result_list            = []
    len_of_values          = len(dataframe[column])
    value_list, count_list = np.unique(dataframe[column], return_counts=True)  # To have value and the counts of them.

    for value in value_list:
        subdata = dataframe.loc[dataframe[column] == value]  # Filtering the rows based on current value.
        result  = entropy_gini_calc(subdata, criterion)      # Having the result for current value.
        result_list.append(result)                           # Appending the result to the result_list

    result = 0.
    index  = 0
    for count in count_list:
        result = result + (count_list[index] / len_of_values) * result_list[index]  # Calculating the weighted result.
        index += 1

    return result


if __name__ == '__main__':

    df = pd.read_csv('heart_summary.csv')        # Reading the csv file.
    df = df.apply(LabelEncoder().fit_transform)  # Label Encoding

    print("Entropy of overall: ", entropy_gini_calc(df, 'entropy'))
    print("Gini of overall:"    , entropy_gini_calc(df, 'gini'))
    print("----------------------------------------")
    print("Entropy of age:"     , split_calc(df, 'age', 'entropy'))
    print("Gini of age"         , split_calc(df, 'age', 'gini'))
    print("----------------------------------------")
    print("Entropy of cp:"      , split_calc(df, 'cp', 'entropy'))
    print("Gini of cp:"         , split_calc(df, 'cp', 'gini'))
    print("----------------------------------------")
    print("Entropy of trestbps:", split_calc(df, 'trestbps', 'entropy'))
    print("Gini of trestbps:"   , split_calc(df, 'trestbps', 'gini'))
