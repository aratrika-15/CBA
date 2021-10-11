import pandas as pd
import numpy as np
from feature_engine.discretisation import DecisionTreeDiscretiser

def read_dataset(name):
    """
    Specifically reads UCI machine learning datasets in the format <name>.data and <name>.scheme saved in the datasets/ folder.
    """
    with open("datasets/{}.names".format(name), "r") as fileobj:
        schema = fileobj.readlines()
        schema = [x[:-1].split(',') for x in schema]  # last character of x is always \n, gets ignored
    
    df = pd.read_csv("datasets/{}.data".format(name), names=schema[0], na_values='?')
    return df, schema


def clean_missing_values(df, schema, nan_threshold=0.5):
    """
    Cleans missing values of a dataframe df with the given schema.
    Numerical values are replaced by mean.
    Categorical values are replaced by mode.

    nan_threshold is the maximum fraction of NaN values allowed in a column. Beyond this threshold, the column is discarded.
    """

    discardCols = set()
    clean_df = df.copy()
    clean_schema = schema.copy()
    for i in range(len(schema[0])):
        colname = schema[0][i]
        coltype = schema[1][i]
        nan_rate = sum(df[colname].isna())/len(df[colname])
        if nan_rate > nan_threshold and coltype != 'label':  # too many NaNs, discard the column
            discardCols.add(i)
        elif nan_rate > 0:
            if coltype == 'numerical':
                clean_df[colname] = clean_df[colname].fillna(clean_df[colname].mean())
            elif coltype == 'categorical':
                clean_df[colname] = clean_df[colname].fillna(clean_df[colname].mode()[0])
            else:  # label has missing values - should not happen, but use mode just in case
                clean_df[colname] = clean_df[colname].fillna(clean_df[colname].mode()[0])
    
    # dropping discardable columns
    if len(discardCols):
        discardColNames = [schema[0][x] for x in discardCols]
        # updating clean_df
        clean_df.drop(columns=discardColNames, inplace=True)
        # updating schema
        clean_schema[0] = [i for j, i in enumerate(schema[0]) if j not in discardCols]
        clean_schema[1] = [i for j, i in enumerate(schema[1]) if j not in discardCols]

    return clean_df, clean_schema


def label_encode(df, schema):
    """
    Uses label encoding on a dataframe's categorical attributes.
    IMPORTANT: Ignores categorical variables that are numeric, to avoid interfering with previously encoded categories like 0/1 or 1-10 scales.
    """

    clean_df = df.copy()
    for i in range(len(schema[0])):
        colname = schema[0][i]
        coltype = schema[1][i]
        if coltype == 'categorical' and not np.issubdtype(clean_df[colname].dtype, np.number):
            clean_df[colname] = clean_df[colname].astype('category')
            clean_df[colname] = clean_df[colname].cat.codes
    return clean_df


def discretize(df, schema):
    """
    Uses a decision tree to discretize the dataset's numerical attributes. Gini index is used as the impurity metric.
    """

    toDiscretize = [schema[0][i] for i in range(len(schema[0])) if schema[1][i] == 'numerical']
    label = schema[0][schema[1].index('label')]
    print("Discretizing:", toDiscretize)
    print('Using label:', label)
    if len(toDiscretize) == 0:
        return df
    disc = DecisionTreeDiscretiser(scoring='accuracy',
                                   variables=toDiscretize,
                                   regression=False,
                                   param_grid={'max_depth': [2,3,4], 'min_samples_leaf':[10,4,2], 'criterion':['gini']},
                                   random_state=42069)
    
    disc.fit(df, df[label])
    disc_df = disc.transform(df.copy())

    # representing the discretized categories as label encodings in sorted order
    for colname in toDiscretize:
        # print(disc_df[colname].unique())
        labelSet = set()
        for i in range(len(disc_df)):
            labelSet.add(disc_df[colname][i])
        # print(labelSet)
        encodingDict = {}
        for i, lab in enumerate(sorted(list(labelSet))):
            encodingDict[lab] = i
        disc_df[colname] = disc_df[colname].map(encodingDict)

    return disc_df


def preprocess_to_array(inp_df, inp_schema):
    df, schema = clean_missing_values(inp_df, inp_schema)
    df = label_encode(df, schema)
    df = discretize(df, schema)
    final_array = df.values.tolist()
    return final_array

if __name__ == "__main__":  # for testing only

    pd.set_option('display.max_columns', None)  # ensures all columns are printed

    # df, schema = read_dataset("horse-colic")  # lots of missing values, mixed bag but all numbers (even categorical)
    # df, schema = read_dataset("facebook")  # a few missing values, mixed bag but all numbers (even categorical)
    # df, schema = read_dataset("tic-tac-toe")  # all categorical non-numeric (x, o, and b)
    df, schema = read_dataset("iris")
    # print(df.describe())
    # print(df.info())

    # df, schema = clean_missing_values(df, schema)
    # print(df.describe())
    # print(df.info())
    # print('\n')
    # print(df.head())
    # for i in range(len(schema[0])):
    #     print(schema[0][i], schema[1][i], sep='\t')

    # df = label_encode(df, schema)
    # print(df.describe())
    # print(df.info())
    # print('\n')
    # print(df.head())

    # df = discretize(df, schema)
    # print(df.describe())
    # print(df.info())
    # print('\n')
    # print(df.head())

    # final_array = df.values.tolist()

    final_array = preprocess_to_array(df, schema)
    print(final_array[:5])