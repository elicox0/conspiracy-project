import os
import numpy as np
import pandas as pd

os.chdir('data')
df = pd.read_csv("conspiracy_theories_data_orig.csv")

# Only NaN values are in "major" column, so no other cleaning is necessary
# Benefit of working with survey data as opposed to data collected using messier methods
# TODO: check for survey responses that don't make sense (answered just the default answer for all 
# questions); these should be thrown out

# Measure for General Conspiracy Belief. Normalized average of responses to questions 1-15 of survey
df['GCB'] = df[['Q'+str(i) for i in range(1, 16)]].mean(axis=1) / 5

# The survey asked participants what words they knew. Columns VCL6, VCL9, VCL12 were not real words, and were included in 
# order to perform a validity check
df['validity'] = df[['VCL6', 'VCL9', 'VCL12']].mean(axis=1)
df['vocabulary_knowledge'] = df[['VCL' + str(i) for i in [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16]]].mean(axis=1)

#I split up every instance of "major" to a category: HUM (Humanities), BUS (business/law), ART, STEM, and OTHER. 
#This block creates a one-hot encoding for each of these.
names = ["STEM", "HUM", "BUS", "OTHER", "ART"]
for f in names:
    with open(f"{name}.txt") as fin:
        majors = [i[:-2] for i in fin.readlines()]
    func = np.vectorize(lambda x: int(x in majors))
    df[name] = 1
    df[name] = df.major.apply(func)

