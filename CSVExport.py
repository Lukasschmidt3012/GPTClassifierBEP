import json
from pymongo import MongoClient
import pandas as pd
import re
from collections import Counter
import numpy as np


client = MongoClient('mongodb://localhost:27017/')
db = client['New']
collection = db['New']



###################################

def db_query(disorder, samples):
    pipeline = [
    {
        '$match': {
            'class': disorder  # Filter on the class
        }
    },
    {
        '$group': {
            '_id': '$user_id',  # Group by the user ID
            'tweet': {'$push': '$tweet'}  # Collect the tweets for each user
        }
    },
    {
        '$project': {
            'sampled_tweets': {'$slice': ['$tweet', 45]}  # Select 45 Tweets without replacement
        }
    }
]
    result = db.New.aggregate(pipeline) # Contains all the grouped tweets by user, aggregation is already done in this part
    return result
# Each json object returned from the DB contains already 100 randomly sampled tweets, no seed is given
# Therefore reproducibility is an issue right now 


def db_query_limited(disorder:str, samples:int):
    pipeline = [
        {
            '$match': {
                'class': disorder  # Filter on the class
            }
        },
        {
            '$group': {
                '_id': '$user_id',  # Group by the user ID
                'tweet': {'$push': '$tweet'}  # Collect the tweets for each user
            }
        },
        {
            '$project': {
                'sampled_tweets': {'$slice': ['$tweet', samples]}  # Select 'samples' number of Tweets without replacement
            }
        },
        {
            '$limit': samples  # Limit the total number of entries returned to 100
        }
    ]
    result = db.New.aggregate(pipeline)  # Contains all the grouped tweets by user, aggregation is already done in this part
    return result



###################################


def create_empty_frame():
    data = {
    '_id': np.zeros(20000),
    'sampled_tweets': np.zeros(20000),
    'ground_truth': np.zeros(20000)
}
    export_frame = pd.DataFrame(data)
    return export_frame




###################################

def create_df_from_db(disorder : str, samples:int):
    result = db_query(disorder, samples)
    export_frame = create_empty_frame()
    counter = 0
    for document in result:
        export_frame.iloc[counter, 0] = document["_id"]
        export_frame.iloc[counter, 1] = ';'.join(document["sampled_tweets"])
        export_frame.iloc[counter, 2] = disorder
        counter += 1

    export_frame= export_frame[export_frame['ground_truth'] != 0.00] # Delete all unnecessary entries created with numpy

    return export_frame

def create_df_from_db_limited(disorder : str, samples:int):
    result = db_query_limited(disorder, samples)
    export_frame = create_empty_frame()
    counter = 0
    for document in result:
        export_frame.iloc[counter, 0] = document["_id"]
        export_frame.iloc[counter, 1] = ';'.join(document["sampled_tweets"])
        export_frame.iloc[counter, 2] = disorder
        counter += 1

    export_frame= export_frame[export_frame['ground_truth'] != 0.00] # Delete all unnecessary entries created with numpy

    return export_frame



###################################

def multi_replace(sampled_tweets):
  words_to_replace = {'adhd': '', 
                    'ptsd': '',
                    'eating disorder': '',
                    'autism' : "", 
                    "autistic" : "",
                    "bipolar" : "",
                    "bipo": "",
                    "eating": "",
                    "depression" : "",
                    "depressed" : "",
                    "ocd" : "",
                    "schizophrenia" : "",
                    "schizo" :"",
                    "post traumatic stress disorder" : "",
                    "eating" : "",
                    "diagnosed" : "",
                    "panicattack" : ""
                   }
  for old, new in words_to_replace.items():
    sampled_tweets = sampled_tweets.replace(old, new)
  return sampled_tweets


###################################


def clean_data(export_frame):
    export_frame['sampled_tweets'] = export_frame['sampled_tweets'].apply(lambda x: multi_replace(x))
    return export_frame



###################################

#export_frame = create_df_from_db("ADHD", 10 )
#export = clean_data(export_frame)
#export.to_csv("ADHDExport.csv")
