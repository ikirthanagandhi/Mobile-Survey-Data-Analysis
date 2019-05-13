# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:42:35 2019

@author: kirth
"""

###########################################################################
###CODE FOR EDA
###########################################################################
# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

customers_df = pd.read_excel('FINALEXAM_Mobile_App_Survey_data.xlsx')

###############################################################################
# Exploratory Data Analysis
###############################################################################

customers_df.info()


customers_desc = customers_df.describe(percentiles = [0.01,
                                                      0.05,
                                                      0.10,
                                                      0.25,
                                                      0.50,
                                                      0.75,
                                                      0.90,
                                                      0.95,
                                                      0.99]).round(2)



customers_desc.loc[['min',
                    '1%',
                    '5%',
                    '10%',
                    '25%',
                    'mean',
                    '50%',
                    '75%',
                    '90%',
                    '95%',
                    '99%',
                    'max'], :]




# Checking class balances for behavioral questions
print(customers_df['q12'].value_counts())
print(customers_df['q11'].value_counts())

customers_df.info()


# Viewing the first few rows of the data
customers_df.head(n = 5)

########################
# Histograms
########################

# Plotting q1, q57 (to see age and gender distriution of the survey)
fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(2, 1, 1)
sns.distplot(a = customers_df['q57'],
             hist = True,
             kde = True,
             color = 'blue')



plt.subplot(2, 1, 2)
sns.distplot(a = customers_df['q1'],
             hist = True,
             kde = True,
             color = 'red')


plt.show()

###########################################################################
###MODEL CODE
###########################################################################

########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2:-12 ]
customer_features_reduced.head()

customers_df.info()

########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()

scaler.fit(customer_features_reduced)

X_scaled_reduced = scaler.transform(customer_features_reduced)

########################
# Step 3: Run PCA without limiting the number of components
########################

customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)

########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca_reduced.n_components_)

plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')

plt.title('Mobile App Research Survey')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()

########################
# Step 5: Run PCA again based on the desired number of components
########################

customer_pca_reduced = PCA(n_components = 3,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)

########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))

factor_loadings_df = factor_loadings_df.set_index(customers_df.columns[2:-12])

print(factor_loadings_df)


factor_loadings_df.to_excel('practice_factor_loadings.xlsx')
########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)

X_pca_df = pd.DataFrame(X_pca_reduced)

########################
# Step 8: Rename your principal components and reattach demographic information
########################

X_pca_df.columns = ['Enthusiasts', 'Influencers', 'Old-fashioned']

final_pca_df = pd.concat([customers_df.loc[ : , ['q1','q48', 'q49', 'q50r1', 'q50r2', 'q50r3', 'q50r4', 'q50r5', 'q54', 'q55', 'q56', 'q57']] , X_pca_df], axis = 1)

########################
# Step 9: Analyze in more detail
########################

# Renaming age
age = {1 : 'under 18',
                 2 : '18-24',
                 3 : '25-29',
                 4 : '30-34',
                 5 : '35-39',
                 6 : '40-44',
                 7 : '45-49',
                 8 : '50-54',
                 9 : '55-59',
                 10 : '60-64',
                 11 : '65 or over'}

final_pca_df['q1'].replace(age, inplace = True)

# Renaming education
education = {1 : 'Some high school',
                2 : 'High school graduate',
                3 : 'Some college',
                4 : 'College graduate',
                5 : 'Some post-graduate studies',
                6 : 'Post graduate degree'}

final_pca_df['q48'].replace(education, inplace = True)

# Renaming marital_status
marital_status = {1 : 'Married',
                2 : 'Single',
                3 : 'Single with a partner',
                4 : 'Separated/Widowed/Divorced'}

final_pca_df['q49'].replace(marital_status, inplace = True)

# Renaming race
race = {1 : 'White or Caucasian',
                2 : 'Black or African American',
                3 : 'Asian',
                4 : 'Native Hawaiian or Other Pacific Islander',
                5 : 'American Indian or Alaska Native',
                6 : 'Other race'}

final_pca_df['q54'].replace(race, inplace = True)

# Renaming ethnicity
ethnicity = {1 : 'Yes',
                2 : 'No'}

final_pca_df['q55'].replace(ethnicity, inplace = True)

# Renaming annual_income
annual_income = {1 : 'Under $10,000',
                2 : '$10,000-$14,999',
                3 : '$15,000-$19,999',
                4 : '$20,000-$29,999',
                5 : '$30,000-$39,999',
                6 : '$40,000-$49,999',
                7 : '$50,000-$59,999',
                8 : '$60,000-$69,999',
                9 : '$70,000-$79,999',
                10 : '$80,000-$89,999',
                11 : '$90,000-$99,999',
                12 : '$100,000-$124,999',
                13 : '$125,000-$149,999',
                14 : '$150,000 and over'}

final_pca_df['q56'].replace(annual_income, inplace = True)

# Renaming gender
gender = {1 : 'Male',
                2 : 'Female'}

final_pca_df['q57'].replace(gender, inplace = True)

final_pca_df.to_excel('pca_factor.xlsx')

# Analyzing by age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Enthusiasts',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Influencers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Old-fashioned',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Analyzing by education
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Enthusiasts',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Influencers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Old-fashioned',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Analyzing by marital_status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Enthusiasts',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Influencers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Old-fashioned',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Analyzing by race
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'Enthusiasts',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'Influencers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'Old-fashioned',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Analyzing by ethnicity
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'Enthusiasts',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'Influencers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'Old-fashioned',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Analyzing by annual_income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Enthusiasts',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Influencers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y =  'Old-fashioned',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# Analyzing by gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Enthusiasts',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Influencers',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Old-fashioned',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


###########################################
###Clustering
############################################


###############################################################################
# Cluster Analysis One More Time!!!
###############################################################################

from sklearn.cluster import KMeans # k-means clustering


########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2:-12 ]



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k = KMeans(n_clusters = 5,
                      random_state = 508)


customers_k.fit(X_scaled_reduced)


customers_kmeans_clusters = pd.DataFrame({'cluster': customers_k.labels_})


print(customers_kmeans_clusters.iloc[: , 0].value_counts())



########################
# Step 4: Analyze cluster centers
########################

centroids = customers_k.cluster_centers_


centroids_df = pd.DataFrame(centroids)



# Renaming columns
centroids_df.columns = customer_features_reduced.columns


print(centroids_df)


# Sending data to Excel
centroids_df.to_excel('survey_k4_clusters.xlsx')



########################
# Step 5: Analyze cluster memberships
########################


X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)


X_scaled_reduced_df.columns = customer_features_reduced.columns


clusters_df = pd.concat([customers_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)


print(clusters_df)



########################
# Step 6: Reattach demographic information 
########################
final_clusters_df = pd.concat([customers_df.loc[ : , ['q1','q48', 'q49', 'q50r1', 'q50r2', 'q50r3', 'q50r4', 'q50r5', 'q54', 'q55', 'q56', 'q57']] , clusters_df], axis = 1)

print(final_clusters_df)

final_clusters_df.to_excel('cluster_after.xlsx')

###############################################################################
# Combining PCA and Clustering!!!
###############################################################################

########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))

print(pd.np.var(X_pca_df))

########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 4,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())


########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Enthusiasts', 'Influencers', 'Old-fashioned']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('survey_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

survey_clust_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(survey_clust_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat([customers_df.loc[ : , ['q1','q48', 'q49', 'q50r1', 'q50r2', 'q50r3', 'q50r4', 'q50r5', 'q54', 'q55', 'q56', 'q57']],
                                survey_clust_pca_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))



########################
# Step 7: Analyze in more detail 
########################

# Renaming age
age = {1 : 'under 18',
                 2 : '18-24',
                 3 : '25-29',
                 4 : '30-34',
                 5 : '35-39',
                 6 : '40-44',
                 7 : '45-49',
                 8 : '50-54',
                 9 : '55-59',
                 10 : '60-64',
                 12 : '65 or over'}

final_clusters_df['q1'].replace(age, inplace = True)

# Renaming education
education = {1 : 'Some high school',
                2 : 'High school graduate',
                3 : 'Some college',
                4 : 'College graduate',
                5 : 'Some post-graduate studies',
                6 : 'Post graduate degree'}

final_clusters_df['q48'].replace(education, inplace = True)

# Renaming marital_status
marital_status = {1 : 'Married',
                2 : 'Single',
                3 : 'Single with a partner',
                4 : 'Separated/Widowed/Divorced'}

final_clusters_df['q49'].replace(marital_status, inplace = True)

# Renaming race
race = {1 : 'White or Caucasian',
                2 : 'Black or African American',
                3 : 'Asian',
                4 : 'Native Hawaiian or Other Pacific Islander',
                5 : 'American Indian or Alaska Native',
                6 : 'Other race'}

final_clusters_df['q54'].replace(race, inplace = True)

# Renaming ethnicity
ethnicity = {1 : 'Yes',
                2 : 'No'}

final_clusters_df['q55'].replace(ethnicity, inplace = True)

# Renaming annual_income
annual_income = {1 : 'Under $10,000',
                2 : '$10,000-$14,999',
                3 : '$15,000-$19,999',
                4 : '$20,000-$29,999',
                5 : '$30,000-$39,999',
                6 : '$40,000-$49,999',
                7 : '$50,000-$59,999',
                8 : '$60,000-$69,999',
                9 : '$70,000-$79,999',
                10 : '$80,000-$89,999',
                11 : '$90,000-$99,999',
                12 : '$100,000-$124,999',
                13 : '$125,000-$149,999',
                14 : '$150,000 and over'}

final_clusters_df['q56'].replace(annual_income, inplace = True)


# Renaming gender
gender = {1 : 'Male',
                2 : 'Female'}

final_clusters_df['q57'].replace(gender, inplace = True)

# Adding a productivity step
data_df = final_pca_clust_df

########################
# Age
########################

# Enthusiasts
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'Enthusiasts',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()



# Influencers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'Influencers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Old-fashioned
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'Old-fashioned',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()


########################
# Education
########################

# Enthusiasts
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y = 'Enthusiasts',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# Influencers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y = 'Influencers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Old-fashioned
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y = 'Old-fashioned',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()


########################
# Marital Status 
########################

# Enthusiasts
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'Enthusiasts',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# Influencers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'Influencers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Old-fashioned
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'Old-fashioned',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()


########################
# Race
########################

# Enthusiasts
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y = 'Enthusiasts',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# Influencers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y = 'Influencers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Old-fashioned
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y = 'Old-fashioned',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()



########################
# Ethnicity 
########################

# Enthusiasts
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y = 'Enthusiasts',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# Influencers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y = 'Influencers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Old-fashioned
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y = 'Old-fashioned',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()


########################
# Income
########################

# Enthusiasts
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'Enthusiasts',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# Influencers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'Influencers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Old-fashioned
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q56',
            y = 'Old-fashioned',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()


########################
# Gender 
########################

# Enthusiasts
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'Enthusiasts',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()


# Influencers
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'Influencers',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()



# Old-fashioned
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y = 'Old-fashioned',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

