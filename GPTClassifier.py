#GPT Classifier
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





def gpt_classifier_ft(testing, model_id):
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose,1 label should be returned as output, no explanation is needed. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text"
    
    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        # Binary model : ft:gpt-3.5-turbo-0613:personal::8gZJqlav
        resulter = client.chat.completions.create(
            
            model= model_id,
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=20,
            top_p = 1
            )
    
        testing["GPT35Finetuned"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing
    



def gpt_classifier_ft_temp(testing, model_id):
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose,1 label should be returned as output, no explanation is needed. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text"
    
    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        # Binary model : ft:gpt-3.5-turbo-0613:personal::8gZJqlav
        resulter = client.chat.completions.create(
            
            model= model_id,
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=20,
            temperature = 0.0
            )
    
        testing["GPT35Finetuned_temp"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing
    


def gpt_classifier_ft_binary(testing, model_id):
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify whether a person has a psychological disorder or does not have one, this task is based on text and 2 labels can be chosen from : "No psychological disorder", "psychological disorder".
    You can only return 1 label, no explanation is needed. This is an example of the input and output format:
    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
    Label ; No psychological disorder .
    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
    This is the text for you to classify : 
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict whether a person has a psychological disorder or not based on text."
    
    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        # Binary model : ft:gpt-3.5-turbo-0613:personal::8gZJqlav
        resulter = client.chat.completions.create(
            
            model= model_id,
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=20,
            temperature = 0.0
            )
    
        testing["GPT35Finetuned_Binary"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing







def gpt_classifier_turbo_binary(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify whether a person has a psychological disorder or does not have one, this task is based on text and 2 labels can be chosen from : "No psychological disorder", "psychological disorder".
    You can only return 1 label, no explanation is needed. This is an example of the input and output format:
    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
    Label ; No psychological disorder .
    Input ; I cant pin point whats wrong i just know i dont feel good ; Lately Ive not been very well, Ive been sleeping alot lately so i know  has me in a headlock but Im trying my best, smiling through it and what not
    Label ; Psychological disorder
    This is the text for you to classify : 
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict whether a person has a psychological disorder or not based on text. "

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        resulter = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            top_p = 1
            )
       
    
        testing["GPT35TurboBaseTopP1_Binary"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing







def gpt_classifier_turbo_binary(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify whether a person has a psychological disorder or does not have one, this task is based on text and 2 labels can be chosen from : "No psychological disorder", "psychological disorder".
    You can only return 1 label, no explanation is needed. This is an example of the input and output format:
    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
    Label ; No psychological disorder .
    Input ; I cant pin point whats wrong i just know i dont feel good ; Lately Ive not been very well, Ive been sleeping alot lately so i know  has me in a headlock but Im trying my best, smiling through it and what not
    Label ; Psychological disorder
    This is the text for you to classify : 
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict whether a person has a psychological disorder or not based on text. "

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        resulter = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            top_p = 1
            )
       
    
        testing["GPT35TurboBaseTopP1_Binary"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing





def gpt_classifier_turbo_binary_temp(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify whether a person has a psychological disorder or does not have one, this task is based on text and 2 labels can be chosen from : "No psychological disorder", "psychological disorder".
    You can only return 1 label, no explanation is needed. This is an example of the input and output format:
    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
    Label ; No psychological disorder .
    Input ; I cant pin point whats wrong i just know i dont feel good ; Lately Ive not been very well, Ive been sleeping alot lately so i know  has me in a headlock but Im trying my best, smiling through it and what not
    Label ; Psychological disorder
    This is the text for you to classify : 
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict whether a person has a psychological disorder or not based on text. "

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        resulter = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            temperature = 0.0
            )
       
    
        testing["GPT35TurboBaseTemp0_Binary"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing




def gpt_classifier_4(testing):
    openai.api_key = os.environ["API"]
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose,1 label should be returned as output, no explanation is needed. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text." 

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        resulter = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            temperature=0.0
            )
    
        testing["Prediction_GPT4_Temp0"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing






def gpt_classifier_turbo_multi_temp0(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose,1 label should be returned as output, no explanation is needed. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text. "

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        resulter = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            temperature = 0.0
            )
       
    
        testing["GPT35TurboBaseTemp0_Multi"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing







def gpt_classifier_turbo_multi_top(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose,1 label should be returned as output, no explanation is needed. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text. "

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        resulter = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            top_p = 1
            )
       
    
        testing["GPT35TurboBaseTopPMulti"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing
















































def gpt_reasoning(testing, model_id):
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    Explain why you chose to classify this user as : 
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text"
    
    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        # Binary model : ft:gpt-3.5-turbo-0613:personal::8gZJqlav
        resulter = client.chat.completions.create(
            
            model="ft:gpt-3.5-turbo-1106:personal::8hmyAfAD",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=300,
            top_p=1
            
            )
    
        testing["Reasoning"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing





def gpt_reasoning4(testing):
    openai.api_key = os.environ["API"]
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose,1 label should be returned as output, you need to explain why you chose the specific disorder. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict whether a person has a psychological disorder or not based on text. "

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        resulter = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=300,
            temperature=0.0
            )
    
        testing["Reasoning_GPT4"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing





def gpt_classifier(testing, model_id):
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose,1 label should be returned as output, you need to explain why you chose the specific disorder. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text"
    
    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        # Binary model : ft:gpt-3.5-turbo-0613:personal::8gZJqlav
        resulter = client.chat.completions.create(
            
            model= model_id,
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=300,
            top_p=1
            
            )
    
        testing["Reasoning"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing
    


def gpt_classifier_turbo(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify whether a person has a psychological disorder or does not have one, this task is based on text and 2 labels can be chosen from : "No psychological disorder", "psychological disorder".
    You can only return 1 label, no explanation is needed. This is an example of the input and output format:
    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
    Label ; No psychological disorder .
    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
    This is the text for you to classify : 
    """

    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict whether a person has a psychological disorder or not based on text. "

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        # OLD FT MODEL : ft:gpt-3.5-turbo-0613:nlptn::8Tas44EK
        # Newest model 2024 ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # ft:gpt-3.5-turbo-0613:nlptn::8VI27d34
    #ft:gpt-3.5-turbo-0613:nlptn::8ccI2wss
        # Even newer model 2024 : ft:gpt-3.5-turbo-0613:nlptn::8dPsuLbD
        resulter = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            top_p = 1
            )
       
    
        testing["GPT35TurboBaseTopP1"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing



#def gpt_classifier_4(testing):
#    openai.api_key = os.environ["API"]
## Base Prompt and System Setting are needed for the GPT Classification for Multi Class
#    base_prompt = """
#    Classify whether a person has a psychological disorder or does not have one, this task is based on text and 2 labels can be chosen from : "No psychological disorder", "psychological disorder".
#    You can only return 1 label, no explanation is needed. This is an example of the input and output format:
#    Input ; WordPress Users Need to Watch Out for Fake Copyright Infringement Warnings HTTPURL #scam #wordpress HTTPURL"";""Linked - Ten Red Flags That Scream 'Don't Take This Job Offer' HTTPURL HTTPURL"";""Report: 94% of women in tech say they're held to a higher standard than men HTTPURL"";
#    Label ; No psychological disorder .
#    Input ; I cant pin point whats wrong i just know i dont feel good ; Lately Ive not been very well, Ive been sleeping alot lately so i know  has me in a headlock but Im trying my best, smiling through it and what not
#    Label ; Psychological disorder
#    This is the text for you to classify : 
#    """

#    system_setting = "You're a classifcation bot for psychological disorders, your task is to accurately predict whether a person has a psychological disorder or not based on text. "

#    for index, row in testing.iterrows():
#        user_prompt = "" + base_prompt
#        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
#        print("Start GPT")
#        resulter = openai.chat.completions.create(
#            model="gpt-4-1106-preview",
#            messages=[
#                   {"role": "system", "content": system_setting},
#                    {"role": "user", "content": user_prompt}
#                    ],
#            max_tokens=50,
#            temperature=0.0
#            )
    
#        testing["Prediction_GPT4"][index] = resulter.choices[0].message.content
    
#        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

#    return testing







def gpt_classifier_4_multi(testing):
    openai.api_key = os.environ["API"]
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose, only 1 label should be returned as output, no explanation needed. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting =  "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text"
 

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        resulter = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            temperature=0.0
            )
    
        testing["Prediction_GPT4_Multi_Temp0"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing



def gpt_classifier_35turbo_multi(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : "ADHD", "AUTISM", "PTSD", "EATING DISORDER", "BIPOLAR" , "SCHIZOPHRENIA", "OCD".
    . What would you choose, only 1 label should be returned as output, no explanation needed. Here's an example of how the output must look like 
    Input ; "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Output ; ADHD
    Input ; "Ahh. It's that time of year again, when my universities system is riddled with errors, the timetable is late, you can't enrol without having to submit enquiries which take days for a response, and it is all around pure stress until the actual course starts."
    Output ; OCD
    This is the text : 
    Input ; "A keto diet and intermittent fasting, gentle exercise, nature, and lots of meditation and self-compassion have helped me a lot."
    Output ; BIPOLAR
    Input ; "Just wish it didn't give me , though the feeling of being too slowed down makes total sense now"
    Output ; AUTISM
    """

    system_setting =  "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text"
 

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        resulter = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                   {"role": "system", "content": system_setting},
                    {"role": "user", "content": user_prompt}
                    ],
            max_tokens=50,
            top_p = 1
            )
    
        testing["Prediction_GPT35Turbo_Multi"][index] = resulter.choices[0].message.content
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].message.content))

    return testing


def gpt_classifier_35turboinstr_multi(testing):
    openai.api_key = os.environ["API"]
    load_dotenv()
    api_key  = os.getenv("API")
    client = OpenAI(api_key = api_key)
# Base Prompt and System Setting are needed for the GPT Classification for Multi Class
    base_prompt = """
    Classify a person's psychological disorders based on text and label using these labels : [ADHD, AUTISM, PTSD, EATING DISORDER, BIPOLAR , SCHIZOPHRENIA]
    . What would you choose, only 1 label should be returned as output, no explanation needed. Here's an example of how the output must look like 
    Text : "@USER @USER Words that reinforce &amp; express a system of social &amp; economic oppression cause actual harm. It's just not immediate and visible, so privileged people like to say it's not the same. Courts recognize that emotional abuse is real; this is on a societal scale."
    Labels : ADHD, AUTISM, PTSD, EATING DISORDER, BIPOLAR , SCHIZOPHRENIA
    This is the text : 
    """

    system_setting =  "You're a classifcation bot for psychological disorders, your task is to accurately predict the psychological disorder of a user based on text"
 

    for index, row in testing.iterrows():
        user_prompt = "" + base_prompt
        user_prompt = user_prompt + str(testing["sampled_tweets"][index])
        print("Start GPT")
        resulter = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt = system_setting + user_prompt,
            max_tokens=50,
            temperature=0.0
            )
    
        testing["Prediction_GPT35TurboINSTR_Multi"][index] = resulter.choices[0].text.replace("\n", "")
    
        print("Finished classifcation for user with id " + str(testing["_id"][index]) + " the result is " +str(resulter.choices[0].text.replace("\n", "")))

    return testing


