# **Heart Attack Analysis **


**GOAL**

- The goal of this exercise is to identify the parameters that influences the heart attack and build a ML model for the prediction of heart attack.


**DATASET**

Dataset can be downloaded from [here](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset).

# About the data

- To predict the chance of Heart attack

## Attributes in the dataset


- **age** - age in years

- **sex** - 1 = male; 0 = female

- **cp** - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic)

- **trtbps** - resting blood pressure

- **chol** - serum cholestoral in mg/dl

- **fbs** - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

- **restecg** - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)

- **thalach** - maximum heart rate achieved

- **exang** - exercise induced angina (1 = yes; 0 = no)

- **oldpeak** - ST depression induced by exercise relative to rest

- **slope** - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)

- **caa** - number of major vessels (0-3) colored by flourosopy

- **thal** - Thalium Stress Test result

- **output** - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < diameter narrowing; Value 1 = > 50% diameter narrowing)

**WHAT I HAD DONE**

- Importing Libraries
- Missing Value Analysis
- Categoric and Numeric Features
- Standardization
- Box - Swarm - Cat - Correlation Plot Analysis
- Outlier Detection
- Machine Learning Model


**MODELS USED**

-  Logistic Regression
-  Linear SVC
-  Support Vector Machines
-  Random Forest
-  KNN
-  Stochastic Gradient Decentt
-  Decision Tree
-  Gaussian Naive Bayes


**LIBRARIES NEEDED**

- numpy
- pandas
- seaborn
- matplotlib
- scipy.stats
- scikit-learn

**Accuracy of different models used**

By using Logistic Regression I got 
 ```python
    test accuracy score of  Logistic Regression =  93.42105263157895
 ``` 

By using Random Forest Classifier I got 
 ```python
    test accuracy score of Random Forest =  86.8421052631579
 ``` 
 
 By using Decision Tree Classifier I got 
 ```python
    test accuracy score of Decision Tree =  81.57894736842105
 ``` 
 
  By using  Support Vector Machine I got 
 ```python
    test accuracy score of Support Vector Machine =  90.78947368421053
 ``` 
 
  By using  Stochastic Gradient Descentt I got 
 ```python
    test accuracy score of Stochastic Gradient Descentt =  82.89473684210526
 ``` 
 
  By using Linear SVC I got 
 ```python
    test accuracy score of  Linear SVC =  90.78947368421053
 ``` 
 
  By using KNN I got 
 ```python
    test accuracy score of KNN =  85.52631578947368
 ``` 
 
  By using Gaussian Naive Bayes I got 
 ```python
    test accuracy score of Gaussian Naive Bayes =  77.63157894736842
 ``` 
 
 
 **CONCLUSION**
 
- Most of the models are working brilliantly on this dataset after normalising the dataset.
- Performance of **Logistic regression** is better in terms of accuracy as compared to other model.
- Only looking at accuracy as evaluation metrics in this case might be deadly as we need to look for **False Negative**.
- Hence , we are looking at complete classification report , especisally **Recall**



### BY :  Harshita Nayak
