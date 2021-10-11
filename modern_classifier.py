import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import new_preprocessing
from split_dataframe import define_splitting
import time


def print_average_results(accuracies,train_times,test_times):
    print()
    print("Average accuracy: ",sum(accuracies)/len(accuracies))
    print("Average training time: ",sum(train_times)/len(train_times))
    print("Average testing time: ",sum(test_times)/len(test_times))

def evaluate_model(df,model):
    split_point=define_splitting(df)
    accuracies=[]
    train_times=[]
    test_times=[]
    for k in range(len(split_point)-1):
        print("\nRound %d:" % k)
        train_data_1 = df.iloc[:split_point[k],:]
        train_data_2 = df.iloc[split_point[k+1]:,:]
        train_data=pd.concat([train_data_1,train_data_2])
        x_train=train_data.iloc[:,:len(df.columns)-1]
        y_train=train_data.iloc[:,len(df.columns)-1:]
        # print(train_data.shape)
        # print(train_data)

        test_data = df.iloc[split_point[k]:split_point[k+1],:]
        # print(test_data.shape)
        # print(test_data)
        x_test=test_data.iloc[:,:len(df.columns)-1]
        y_test=test_data.iloc[:,len(df.columns)-1:]

        clf = model
        start_time = time.time()
        clf.fit(x_train,y_train.values.ravel())
        end_time= time.time()
        time_for_training=end_time-start_time
        train_times.append(time_for_training)

        start_time=time.time()
        accuracy=clf.score(x_test,y_test)
        end_time=time.time()
        time_for_testing=end_time-start_time
        test_times.append(time_for_testing)

        accuracies.append(accuracy)
        print("Test Accuracy: ",accuracy)
        print("Time for training: ",time_for_training)
        print("Time for testing: ",time_for_testing)

    print_average_results(accuracies,train_times,test_times)
    
    



if __name__ == '__main__':
    dataframe=new_preprocessing.create_dataframe('iris')
    model=DecisionTreeClassifier(random_state=0)
    print("Results of Decision Tree Classifier:")
    evaluate_model(dataframe,model)
    print()
    model=RandomForestClassifier()
    print("Results of Random Forest Classifier:")
    evaluate_model(dataframe,model)
    print()
    model=SVC()
    print("Results of Support Vector Classifier:")
    evaluate_model(dataframe,model)
    model=GaussianNB()
    print("Results of Naive Bayes Classifier:")
    evaluate_model(dataframe,model)

