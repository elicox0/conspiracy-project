#!/usr/bin/env python
# coding: utf-8

# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# In[95]:


df = pd.read_csv("conspiracy_theories_data_orig.csv")
verbose = False
# Only NaN values are in "major" column, so no other cleaning is necessary
# Benefit of working with survey data as opposed to data collected using messier methods
# TODO: check for survey responses that don't make sense (answered just the default answer for all 
# questions); these should be thrown out

# Measure for General Conspiracy Belief. Normalized average of responses to questions 1-15 of survey
df['GCB'] = df[['Q'+str(i) for i in range(1, 16)]].mean(axis=1) / 5
df.drop(columns=['Q'+str(i) for i in range(1, 16)], inplace=True)

# The survey asked participants what words they knew. Columns VCL6, VCL9, VCL12 were not real words, and were included in 
# order to perform a validity check

df['validity'] = df[['VCL6', 'VCL9', 'VCL12']].mean(axis=1)
# Score how many vocab questions the respondent answered correctly. 0 is correct for VCL 6, 9, 12, and 1 is correct for all others.

df['vocabulary_knowledge'] = (df[['VCL' + str(i) for i in [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16]]] 
                              + (1 - df[['VCL' + str(i) for i in [6,9,12]]])).mean(axis=1)

df.drop(columns=['VCL'+str(i) for i in range(1, 17)], inplace=True)

#I split up every instance of "major" to a category: HUM (Humanities), BUS (business/law), ART, STEM, and OTHER. 
#This block creates a one-hot encoding for each of these.
names = ["STEM", "HUM", "BUS", "OTHER", "ART"]
for name in names:
    # For each category, there is a file of strings of majors that should be classified as that category
    # Read in the corresponding file
    tf = open(f"{name}.txt", "r",newline='\n')
    # Grab all the strings in the file
    majors = [i[:-2] for i in tf.readlines()]
    def func(x): # If string is in the list of majors, return a 1, else a 0
        return int(x in majors)
    func = np.vectorize(func)
    df[name] = 1 
    df[name] = df.major.apply(func) # Create  a new column with the one hot encoding for the given category
    
# One hot encode the other features
categorical_columns = ['education','urban', 'gender', 'engnat', 'hand', 'religion', 'orientation','race', 'voted', 'married']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df["constant"] = 1
df.columns
# The columns 'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8',
#        'TIPI9', 'TIPI10', 'age', 'familysize', 'major', 'GCB', 'validity',
#        'vocabulary_knowledge', 'STEM', 'HUM', 'BUS', 'OTHER', 'ART',
#        'education_1', 'education_2', 'education_3', 'education_4', 'urban_1',
#        'urban_2', 'urban_3', 'gender_1', 'gender_2', 'gender_3', 'engnat_1',
#        'engnat_2', 'hand_1', 'hand_2', 'hand_3', 'religion_1', 'religion_2',
#        'religion_3', 'religion_4', 'religion_5', 'religion_6', 'religion_7',
#        'religion_8', 'religion_9', 'religion_10', 'religion_11', 'religion_12',
#        'orientation_1', 'orientation_2', 'orientation_3', 'orientation_4',
#        'orientation_5', 'race_1', 'race_2', 'race_3', 'race_4', 'race_5',
#        'voted_1', 'voted_2', 'married_1', 'married_2', 'married_3',
#        'constant'
# are all fair game for regression. 


# In[75]:


# Are there any rows where the user just answered the same for all relevant questions? 
# Are there any rows that were only partially completed? 
    # For the above two, could look at time to complete survey
# TODO: One-hot encode the rest of the categorical data


# In[81]:


# df["introelapse"].hist()
df["total_time_taken_(mins)"] = (df["introelapse"] + df["testelapse"] + df["surveyelapse"])/60
df["total_survey_time_taken_(mins)"] = (df["testelapse"] + df["surveyelapse"])/60

print("# Surveys that took over an hour to take (including landing pad time)")
print(sum(df["total_time_taken_(mins)"] >= 60))

print("# Surveys that took over an hour to take (excluding landing pad time)")
print(sum(df["total_survey_time_taken_(mins)"] >= 60))

print("# Surveys that spent over an hour on the landing pad")
print(sum(df["introelapse"]/60 >= 60))

if verbose: 
    df["total_time_taken_(mins)"][df["total_time_taken_(mins)"] < 60].hist()
    plt.subplots()
    df["total_survey_time_taken_(mins)"][df["total_survey_time_taken_(mins)"] < 60].hist()
    plt.subplots()
    df["introelapse"][df["introelapse"] < 60].hist()

# Even though these surveys took a lot longer than seems reasonable, there are no clear indications in the 
# subjects' answers that any of these responses should be dropped. 


# In[78]:


if verbose: 
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        display(df[df["introelapse"]/60 >= 60])


# In[79]:


# Looking at the 50 fastest responses
if verbose:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df.sort_values(by="total_time_taken_(mins)")[:50])
    
# Again, none of these look totally wrong. 


# In[80]:


# Did any respondents put the same thing for each question in the GCB inventory? 
print("# Rows with matching entries in columsn Q1, Q2, ..., Q15")
print(sum(df[["Q" + str(i) for i in range(1, 16)]].apply(lambda x: min(x) == max(x), 1)))
if verbose:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df[df[["Q" + str(i) for i in range(1, 16)]].apply(lambda x: min(x) == max(x), 1)])


# In[ ]:




