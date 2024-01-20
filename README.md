# Find-Default-Prediction-of-Credit-Card-Fraud

The Credit Card Fraud Detection project is used to identify whether a new transaction is fraudulent or not by modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. We will use various predictive models to see how accurate they are in detecting whether a transaction is a normal payment or a fraud.
Classification techniques are the promising solutions to detect the fraud and non-fraud transactions. Unfortunately, in a certain condition, classification techniques do not perform well when it comes to huge numbers of differences in data distribution.
# Problem Statement:
A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
We have to build a classification model to predict whether a transaction is fraudulent or not.

# Exploratory Data Analysis (EDA)
Once the data is read into python, we need to explore/clean/filter it before processing it for machine learning It involves adding/deleting few columns or rows, joining some other data, and handling qualitative variables like dates.

Now that we have the data, I wanted to run a few initial comparisons between the three columns - Time, Amount, and Class
# Load Dataset
df = pd.read_csv('creditcard.csv', encoding='latin_1')

We are using the datasets provided by upGrad capstone project. This data set includes all transactions recorded over the course of two days. As described in the dataset, the features are scaled and the names of the features are not shown due to privacy reasons.
The dataset consists of numerical values from the 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28. Furthermore, there is no metadata about the original features provided, so pre-analysis or feature study could not be done.

df.shape

There are 284807 records. The only thing we know is that those columns that are unknown have been scaled already.

df.isnull().sum()

There are no “Null” values, so we don’t have to work on ways to replace values.

print(df['class'].value_counts())
print('\n')
print(df['class'].value_counts(normalize=True))

# Categorical vs Continuous Features
Finding unique values for each column to understand which column is categorical and which one is Continuous
df[['time','amount','class']].nunique()

# Feature Engineering
Feature engineering on Time

Converting time from second to hour

df['time'] = df['time'].apply(lambda sec : (sec/3600))

# Reset the index
df.reset_index(inplace = True , drop = True)

PCA Transformation: The description of the data says that all the features went through a PCA transformation (Dimensionality Reduction technique) except for time and amount.

Scaling: Keep in mind that in order to implement a PCA transformation features need to be previously scaled.

# Splitting data into Training and Testing samples

We don't use the full data for creating the model. Some data is randomly selected and kept aside for checking how good the model is. This is known as Testing Data and the remaining data is called Training data on which the model is built. Typically 70% of data is used as training data and the rest 30% is used as testing data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=101)

# Baseline for models

We will train four types of classifiers and decide which classifier will be more effective in detecting fraud transactions.
Classification Models

Logistic Regression

Decision Trees

Random Forest

Naive Bayes Classifier

 # 1 Logistic Regression
 
 Logistic Regression with imbalanced data

from sklearn.linear_model import LogisticRegression 

logreg = LogisticRegression()
logreg.fit(X_train, y_train) 

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
cnf_matrix

# Class Imbalance
Imbalanced data typically refers to a problem with classification problems where the classes are not represented equally. If one applies classifiers on the dataset, they are likely to predict everything as the majority class. This was often regarded as a problem in learning from highly imbalanced datasets.

Let's Fix the class Imbalance and apply some sampling techniques.

# Under Sampling and Over Sampling
Oversampling and undersampling in data analysis are techniques used to adjust the class distribution of a data set.

Random oversampling duplicates examples from the minority class in the training dataset and can result in overfitting for some models.

Random undersampling deletes examples from the majority class and can result in losing information invaluable to a model.

# Synthetic Minority OverSampling Technique (SMOTE)
In this technique, instead of simply duplicating data from the minority class, we synthesize new data from the minority class. This is a type of data augmentation for tabular data can be very effective. This approach to synthesizing new data is called the Synthetic Minority Oversampling TEchnique, or SMOTE for short.

# Adaptive Synthetic Sampling Method for Imbalanced Data (ADASYN)
ADASYN (Adaptive Synthetic) is an algorithm that generates synthetic data, and its greatest advantages are not copying the same minority data, and generating more data for “harder to learn” examples.

# Performance measures of various classifiers

data = {'Model':names_lst,
       #'Accuracy_Train':accuracy_train_lst,
       'Accuracy_Test':accuracy_test_lst,
       #'AUC_Train':aucs_train_lst,
       'AUC_Test':aucs_test_lst,
       #'PrecisionScore_Train':precision_train_lst,
       'PrecisionScore_Test':precision_test_lst,
       #'RecallScore_Train':recall_train_lst,
       'RecallScore_Test':recall_test_lst,
       #'F1Score_Train':f1_train_lst,
       'F1Score_Test':f1_test_lst}

print("Performance measures of various classifiers: \n")
performance_df = pd.DataFrame(data) 
performance_df.sort_values(['F1Score_Test','RecallScore_Test','AUC_Test'],ascending=False)

# Grid Search
Grid search is the process of performing hyper parameter tuning in order to determine the optimal values for a given model. This is significant as the performance of the entire model is based on the hyper parameter values specified.

A model hyperparameter is a characteristic of a model that is external to the model and whose value cannot be estimated from data. The value of the hyperparameter has to be set before the learning process begins. For example, c in Support Vector Machines, k in k-Nearest Neighbors, the number of hidden layers in Neural Networks.

# Conclusion
We were able to accurately identify fraudulent credit card transactions using a random forest model with oversampling technique. We, therefore, chose the random forest model with oversampling technique as the better model, which obtained highest F1 score of 86.47% on the test set.


