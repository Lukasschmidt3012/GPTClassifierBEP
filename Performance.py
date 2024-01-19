import os
from pymongo import MongoClient
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter
import requests
import re
import openai
import json
import tiktoken 
import numpy as np
from collections import defaultdict
import chime

# Function to calculate specific accuracy for a disorder and model

def accuracy_calculator(df, disorder, model ):
    counts = (df["ground_truth"] == disorder)
    counts = counts.sum()
    mask = (df['ground_truth'] == disorder) & (df[model] == disorder)
    accuracy = mask.sum()/counts
    return accuracy


# Function to calculate all the accuracies of a model for multiclass classification
def accuracy_per_class_multi(df,model):
    disorder_list = ['ADHD', 'AUTISM', 'BIPOLAR', 'PTSD', 'OCD',
       'EATING DISORDER', 'SCHIZOPHRENIA']
    accuracy_frame = pd.DataFrame(columns=disorder_list)
    accuracy_frame[model] = model
    accuracy_frame.loc[0,model] = 0
    accuracy_frame.set_index(model,inplace=True)
    for i in disorder_list:
        print(accuracy_calculator(df,i, model))
        accuracy_frame.loc[0,i] =  accuracy_calculator(df,i, model)

    return accuracy_frame


# Function to calculate the performance measures for multiclass classification
def calculate_performance_multi(testing,model):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['PTSD', 'SCHIZOPHRENIA', 'EATING DISORDER', 'BIPOLAR', 'AUTISM', "OCD",
       'ADHD']
    counter = 0
    for disorder in disorders:
        mask = (testing['ground_truth'] == disorder) & (testing[model] == disorder)
        tp = mask.sum()
        mask = (testing["ground_truth"] != disorder)& (testing[model] == disorder)
        fp = mask.sum()
        all_positives = (testing["ground_truth"] == disorder ).sum()
        all_negatives = (testing["ground_truth"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["ground_truth"] == disorder) & (testing[model]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        performance.loc[counter] = [disorder, precision, recall, f1]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
    return performance

# Function to calculate the overall performance of a model, weighted by the amount of ground truth samples
def calculate_performance_multi_averages(input_df, model):
    multi_df_results = calculate_performance_multi(input_df,model)
    values = input_df["ground_truth"].value_counts()
    percentages = pd.DataFrame(values)
    divide = int(percentages["count"].sum())
    percentages.sort_values(by='ground_truth', ascending=True)
    percentages["count"]  = percentages["count"] / divide
    multi_df_results.index = multi_df_results["Disorder"]
    multi_df_results
    multi_df_results = multi_df_results.drop(columns=["Disorder"])

    for index, row in multi_df_results.iterrows():
        multi_df_results.loc[index] = (multi_df_results.loc[index] * percentages.loc[index][0])
    col_sums = multi_df_results.sum(axis=0)
    sums_df = pd.DataFrame(col_sums, columns=['Total'])
    return sums_df


def accuracy_per_class_multi_average(df,model):
    disorder_list = ['ADHD', 'AUTISM', 'BIPOLAR', 'PTSD', 'OCD',
       'EATING DISORDER', 'SCHIZOPHRENIA']
    values = df.ground_truth.value_counts()
    divider = df.ground_truth.value_counts().sum()
    percentages = values / divider
    accuracy_frame = pd.DataFrame(columns=disorder_list)
    accuracy_frame[model] = model
    accuracy_frame.loc[0,model] = 0
    accuracy_frame.set_index(model,inplace=True)
    counter = 0
    for i in disorder_list:
        accuracy_frame.loc[0,i] =  accuracy_calculator(df,i, model) *percentages[counter]
        counter += 1
    return accuracy_frame.sum(axis=1)
