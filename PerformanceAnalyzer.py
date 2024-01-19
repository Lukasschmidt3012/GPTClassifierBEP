#Performance Analyzer



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



def calculate_performance_binary_df(testing,model):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['no psychological disorder', 'psychological disorder']
    counter = 0
    for disorder in disorders:
        mask = (testing['binary_disorder'] == disorder) & (testing[model] == disorder)
        tp = mask.sum()
        mask = (testing["binary_disorder"] != disorder)& (testing[model] == disorder)
        fp = mask.sum()
        all_positives = (testing["binary_disorder"] == disorder ).sum()
        all_negatives = (testing["binary_disorder"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["binary_disorder"] == disorder) & (testing[model]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance



def calculate_performance_binary(testing, model):
    #mapping = {'PTSD': 'psychological disorder', 
    #       'SCHIZOPHRENIA': 'psychological disorder',
    #       'EATING DISORDER': 'psychological disorder',
    #       'BIPOLAR': 'psychological disorder', 
    #       'AUTISM': 'psychological disorder',
    #       'ADHD': 'psychological disorder',
    #       'CONTROL': 'No psychological disorder'}

    #testing["ground_truth"] = testing["ground_truth"].map(mapping)

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['no psychological disorder', 'psychological disorder']
    counter = 0
    for disorder in disorders:
        mask = (testing['binary_disorder'] == disorder) & (testing[model] == disorder)
        tp = mask.sum()
        mask = (testing["binary_disorder"] != disorder)& (testing[model] == disorder)
        fp = mask.sum()
        all_positives = (testing["binary_disorder"] == disorder ).sum()
        all_negatives = (testing["binary_disorder"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["binary_disorder"] == disorder) & (testing[model]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance




def calculate_performance_multi(testing, model):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['PTSD', 'SCHIZOPHRENIA', 'EATING DISORDER', 'BIPOLAR', 'AUTISM','ADHD',"OCD"]
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
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance































def calculate_performance(testing):
    #mapping = {'PTSD': 'psychological disorder', 
    #       'SCHIZOPHRENIA': 'psychological disorder',
    #       'EATING DISORDER': 'psychological disorder',
    #       'BIPOLAR': 'psychological disorder', 
    #       'AUTISM': 'psychological disorder',
    #       'ADHD': 'psychological disorder',
    #       'CONTROL': 'No psychological disorder'}

    #testing["ground_truth"] = testing["ground_truth"].map(mapping)

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['no psychological disorder', 'psychological disorder']
    counter = 0
    for disorder in disorders:
        mask = (testing['ground_truth'] == disorder) & (testing['Prediction'] == disorder)
        tp = mask.sum()
        mask = (testing["ground_truth"] != disorder)& (testing["Prediction"] == disorder)
        fp = mask.sum()
        all_positives = (testing["ground_truth"] == disorder ).sum()
        all_negatives = (testing["ground_truth"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["ground_truth"] == disorder) & (testing["Prediction"]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance



def calculate_performance_gpt3(testing):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['No psychological disorder', 'psychological disorder']
    counter = 0
    for disorder in disorders:
        mask = (testing['ground_truth'] == disorder) & (testing['Prediction_GPT3.5Turbo'] == disorder)
        tp = mask.sum()
        mask = (testing["ground_truth"] != disorder)& (testing["Prediction_GPT3.5Turbo"] == disorder)
        fp = mask.sum()
        all_positives = (testing["ground_truth"] == disorder ).sum()
        all_negatives = (testing["ground_truth"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["ground_truth"] == disorder) & (testing["Prediction_GPT3.5Turbo"]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance



def calculate_performance_gpt4(testing):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['No psychological disorder', 'psychological disorder']
    counter = 0
    for disorder in disorders:
        mask = (testing['ground_truth'] == disorder) & (testing['Prediction_GPT4'] == disorder)
        tp = mask.sum()
        mask = (testing["ground_truth"] != disorder)& (testing["Prediction_GPT4"] == disorder)
        fp = mask.sum()
        all_positives = (testing["ground_truth"] == disorder ).sum()
        all_negatives = (testing["ground_truth"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["ground_truth"] == disorder) & (testing["Prediction_GPT4"]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance


def calculate_performance_gpt4_multi(testing):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['PTSD', 'SCHIZOPHRENIA', 'EATING DISORDER', 'BIPOLAR', 'AUTISM','ADHD',"OCD"]
    counter = 0
    for disorder in disorders:
        mask = (testing['ground_truth'] == disorder) & (testing['Prediction_GPT4_Multi_Temp0'] == disorder)
        tp = mask.sum()
        mask = (testing["ground_truth"] != disorder)& (testing["Prediction_GPT4_Multi_Temp0"] == disorder)
        fp = mask.sum()
        all_positives = (testing["ground_truth"] == disorder ).sum()
        all_negatives = (testing["ground_truth"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["ground_truth"] == disorder) & (testing["Prediction_GPT4_Multi_Temp0"]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance



def calculate_performance_gpt35_multi(testing):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['PTSD', 'SCHIZOPHRENIA', 'EATING DISORDER', 'BIPOLAR', 'AUTISM',
       'ADHD']
    counter = 0
    for disorder in disorders:
        mask = (testing['ground_truth'] == disorder) & (testing['Prediction_GPT35Turbo_Multi'] == disorder)
        tp = mask.sum()
        mask = (testing["ground_truth"] != disorder)& (testing["Prediction_GPT35Turbo_Multi"] == disorder)
        fp = mask.sum()
        all_positives = (testing["ground_truth"] == disorder ).sum()
        all_negatives = (testing["ground_truth"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["ground_truth"] == disorder) & (testing["Prediction_GPT35Turbo_Multi"]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance


def calculate_performance_gpt35INSTR_multi(testing):

    disorders = ["Disorder", 'Precision', 'Recall', 'F1', 'AUC']

    performance = pd.DataFrame(columns=disorders)

    # Calculate performance

    disorders = ['PTSD', 'SCHIZOPHRENIA', 'EATING DISORDER', 'BIPOLAR', 'AUTISM',
       'ADHD']
    counter = 0
    for disorder in disorders:
        mask = (testing['ground_truth'] == disorder) & (testing['Prediction_GPT35TurboINSTR_Multi'] == disorder)
        tp = mask.sum()
        mask = (testing["ground_truth"] != disorder)& (testing["Prediction_GPT35TurboINSTR_Multi"] == disorder)
        fp = mask.sum()
        all_positives = (testing["ground_truth"] == disorder ).sum()
        all_negatives = (testing["ground_truth"] != disorder).sum()
        #tn = (test_dataset["ground_truth"])
        mask = (testing["ground_truth"] == disorder) & (testing["Prediction_GPT35TurboINSTR_Multi"]!= disorder)
        fn = mask.sum()
        precision = tp / (fp + tp)
        recall = tp / (tp+fn) 
        f1 = (2 * precision * recall) / (precision + recall )
        tpr = tp / all_positives
        fpr = fp /all_negatives
        auc = tpr /(tpr+fpr)
        performance.loc[counter] = [disorder, precision, recall, f1,auc]
        counter +=1 
        print(str(disorder))
        print("Precision " +" " +str( tp/(fp+tp))) 
        print("Recall: " + str(tp/(tp+fn)))
        print("F1 " + str(f1))
        print("TPR " + str(tp / all_positives))
        print("FPR " +str(fpr))
        print("AUC " +str(tpr/ (tpr+fpr)))
    return performance


def calculate_averages_gpt35_multi_instr(input_df, model):
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if ':' in x else x)
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if ';' in x else x)
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if '.' in x else x)
    gpt4_multi = calculate_performance_gpt35INSTR_multi(input_df)
    values = input_df['ground_truth'].value_counts()
    percentages = pd.DataFrame(values)

    divide = int(percentages["count"].sum())
    percentages = percentages.sort_values(by='ground_truth', ascending=True)
    gpt4_multi = gpt4_multi.sort_values(by='Disorder', ascending=True)
    summation = percentages["count"].sum()
    percentages["count"]  = percentages["count"] / summation
    gpt4_multi.index = gpt4_multi["Disorder"]
    gpt4_multi = gpt4_multi.drop(columns=['Disorder'])


    for index, row in gpt4_multi.iterrows():
        gpt4_multi.loc[index] = (gpt4_multi.loc[index] * percentages.loc[index][0])
    col_sums = gpt4_multi.sum(axis=0)
    sums_df = pd.DataFrame(col_sums, columns=['Total'])

    return sums_df




    
def calculate_averages_gpt35_multi(input_df, model):
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if ':' in x else x)
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if ';' in x else x)
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if '.' in x else x)
    gpt4_multi = calculate_performance_gpt35_multi(input_df)
    values = input_df['ground_truth'].value_counts()
    percentages = pd.DataFrame(values)

    divide = int(percentages["count"].sum())
    percentages = percentages.sort_values(by='ground_truth', ascending=True)
    gpt4_multi = gpt4_multi.sort_values(by='Disorder', ascending=True)
    summation = percentages["count"].sum()
    percentages["count"]  = percentages["count"] / summation
    gpt4_multi.index = gpt4_multi["Disorder"]
    gpt4_multi = gpt4_multi.drop(columns=['Disorder'])


    for index, row in gpt4_multi.iterrows():
        gpt4_multi.loc[index] = (gpt4_multi.loc[index] * percentages.loc[index][0])
    col_sums = gpt4_multi.sum(axis=0)
    sums_df = pd.DataFrame(col_sums, columns=['Total'])

    return sums_df




def calculate_averages_gpt4_multi(input_df, model):
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if ':' in x else x)
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if ';' in x else x)
    input_df[model] = input_df[model].apply(lambda x: x.split(':')[-1].strip() if '.' in x else x)
    gpt4_multi = calculate_performance_gpt4_multi(input_df)
    values = input_df['ground_truth'].value_counts()
    percentages = pd.DataFrame(values)

    divide = int(percentages["count"].sum())
    percentages = percentages.sort_values(by='ground_truth', ascending=True)
    gpt4_multi = gpt4_multi.sort_values(by='Disorder', ascending=True)
    summation = percentages["count"].sum()
    percentages["count"]  = percentages["count"] / summation
    gpt4_multi.index = gpt4_multi["Disorder"]
    gpt4_multi = gpt4_multi.drop(columns=['Disorder'])


    for index, row in gpt4_multi.iterrows():
        gpt4_multi.loc[index] = (gpt4_multi.loc[index] * percentages.loc[index][0])
    col_sums = gpt4_multi.sum(axis=0)
    sums_df = pd.DataFrame(col_sums, columns=['Total'])

    return sums_df



def accuracy_calculator(df, disorder, model ):
    counts = (df["ground_truth"] == disorder)
    counts = counts.sum()
    mask = (df['ground_truth'] == disorder) & (df[model] == disorder)
    accuracy = mask.sum()/counts
    return accuracy
        
def accuracy_df_multi(accuracy_df):
    disorders = ['PTSD', 'SCHIZOPHRENIA', 'EATING DISORDER', 'BIPOLAR', 'AUTISM', 'ADHD', "OCD"]
    models = ["Prediction_GPT41106_MULTI","Prediction_GPT35Turbo_Multi","Prediction_GPT35TurboINSTR_Multi"]
    accuracy = pd.DataFrame(index=models, columns=disorders)
    for disorder in disorders:
        for model in models:
            accuracy.loc[model][disorder] =  accuracy_calculator(accuracy_df, disorder,model)
    return accuracy

    