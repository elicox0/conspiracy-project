#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

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
for f in names:
    with open(f"{name}.txt") as fin:
        majors = [i[:-2] for i in fin.readlines()]
    func = np.vectorize(lambda x: int(x in majors))
    df[name] = 1
    df[name] = df.major.apply(func)
    
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


# Are there any rows where the user just answered the same for all relevant questions? 
# Are there any rows that were only partially completed? 
    # For the above two, could look at time to complete survey
# TODO: One-hot encode the rest of the categorical data


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


if verbose: 
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        display(df[df["introelapse"]/60 >= 60])


# Looking at the 50 fastest responses
if verbose:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df.sort_values(by="total_time_taken_(mins)")[:50])
    
# Again, none of these look totally wrong. 


# Did any respondents put the same thing for each question in the GCB inventory? 
print("# Rows with matching entries in columsn Q1, Q2, ..., Q15")
print(sum(df[["Q" + str(i) for i in range(1, 16)]].apply(lambda x: min(x) == max(x), 1)))
if verbose:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df[df[["Q" + str(i) for i in range(1, 16)]].apply(lambda x: min(x) == max(x), 1)])

