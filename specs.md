# Team: Laren, Eli, Dallan, Calix
# Project:
Are there some factors that correlate with a person believing or disbelieving in conspiracy theories? Some believe the only uneducated people, or overly educated people have such beliefs, but do these impressions hold any water? (At the very least, among those who would complete an online survey?) For our project we will use online survey data about belief in conspiracy theories to find what factors correlate with a strong belief in conspiracy theories. 
# Guiding Questions: 
We will engineer a feature that will give a single measure of a participant's belief in conspiracy theories using a heuristic weighted average of the participant’s survey responses. identify features from the above list that have the highest correlation coefficient with belief in conspiracy theories. Data include:
   * Basic Demographics
   * Personality
   * Type of area grown up in (Rural, Suburban, Urban)
   * Political Activity (if you voted in the last election)
   * Major in college
# Dataset: 
https://www.kaggle.com/yamqwe/measuring-belief-in-conspiracy-theories/version/1
This dataset consists of survey data taken from over #5 thousand participants. It includes direct         survey information about belief in conspiracy theories as well as demographic and other information about how long the participant took to work each question, vocabulary questions, a brief personality inventory and a few validation questions. Since this is a survey, it is not universally representative, but we expect to be able to draw valuable conclusions.
# Tools:
Regression, data cleaning and possible feature engineering to get our data ready for regression. There is a column asking the participant for their major in college (if they had one) and will need to get cleaned and probably engineered. Features that we will consider engineering include: Time to complete survey (could potentially correlate with intelligence/education level), strength of belief in conspiracy theories (generally), validation, and personality. We can also implement classification algorithms (like a random forest) to predict a person’s belief in conspiracies based on their personal data and demographics.
# Metrics to decide how good answers are
   * For the regression: standard error and R2 metrics
   * Classification algorithm: classic 80-20 train/test split How we will divide work:
# Work Division
Initial data cleaning and feature engineering will be done by Laren and Calix, and regression will be done by Dallan and Eli.. For the write-up, Dallan will do the overall summary of research. Eli the conclusions that we come too from figures and analysis. Laren and Calix will generate and insert our figures and draw conclusions from data. The classification algorithm (as a fairly simple addition) will be implemented by whoever finishes their task most quickly.
