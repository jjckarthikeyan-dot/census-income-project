# Census Income Classification and Customer Segmentation Report

## 1. Executive Summary

In this project, I worked on a retail marketing problem using Census data.

The main goal was to help the client answer two questions:

1. Who is likely to earn more than $50K  
2. How can we group customers into meaningful segments for marketing  

To solve this, I built:
- a classification model to predict income group  
- a segmentation model to understand different types of customers  

One important thing I noticed early is that the number of people earning more than $50K is much smaller than the rest. Because of this, I did not rely only on accuracy and instead focused on metrics like ROC-AUC and recall.

---

## 2. Understanding the Data

The dataset comes from Census surveys conducted in the 1990s. Each row represents a person and includes a weight that shows how much of the population that record represents.

The data includes:
- demographic details like age, gender, and race  
- employment information like occupation and industry  
- financial signals like capital gains and dividends  
- income label (`<=50K` or `>50K`)  

The dataset is clearly imbalanced:
- most people earn less than or equal to $50K  
- only a small group earns more than $50K  

This imbalance affects how the model performs and how we should evaluate it.

---

## 3. Data Preparation

Before building the models, I performed some basic preprocessing:

- replaced `?` values with missing values  
- converted the income label into binary (0 and 1)  
- removed the label column from the input features  
- used the `weight` column during training and evaluation  
- handled missing values using imputation  
- encoded categorical variables using one-hot encoding  

For segmentation, I made a clear decision not to use the income label during training. The goal was to let the data naturally form groups.

---

## 4. Classification Model

### Approach

For the classification task, I used a linear model similar to logistic regression.

I chose this because:
- it works well with large datasets  
- handles one-hot encoded data efficiently  
- easy to understand and explain  

### Evaluation

I used an 80/20 train-test split with stratification.

Main results:
- Accuracy: around 79%  
- Balanced Accuracy: around 84%  
- ROC-AUC: around 0.88  
- PR-AUC: around 0.25  

The model is able to identify most of the high-income individuals, but precision is lower since that group is small.

---

## 5. Business View of Classification

Instead of using only a yes/no prediction, I recommend using the probability output of the model.

For example:
- high probability → target with premium products  
- medium probability → test campaigns  
- low probability → focus on budget offers  

The final decision depends on cost:
- if contacting customers is cheap → focus on recall  
- if expensive → focus on precision  

---

## 6. Segmentation Model

### Approach

For segmentation, I used clustering based on features that are easy to understand from a business point of view:

- age  
- education  
- occupation  
- work status  
- some financial indicators  

I did not use the income label during clustering.

After creating the clusters, I used the income label to understand how each segment behaves.

---

## 7. Segment Insights

After analyzing the segments, I observed:

- some groups are working adults with stable income  
- some groups are younger or just starting their careers  
- one group has a much higher concentration of high-income individuals  
- some groups include people not actively working  

This helps the business decide where to focus different types of campaigns.

---

## 8. Key Observations

Some important things I noticed:

- the dataset is highly imbalanced  
- work-related features are strong indicators of income  
- financial features help identify higher-income groups  
- segmentation gives more insight compared to just prediction  

---

## 9. Limitations and Future Work

There are a few limitations:

- the data is old, so real-world results may differ  
- some features may not be suitable for real marketing use  
- model performance can be improved using more advanced methods  

In the future, I would:
- try models like gradient boosting  
- add better calibration  
- use explainability techniques like feature importance or SHAP  

---

## 10. Final Recommendation

The best approach is to use both models together:

1. use the classifier to estimate income likelihood  
2. use segmentation to understand customer groups  
3. design different campaigns for each segment  
4. test results using A/B testing  

This helps move from generic marketing to more targeted strategies.

---

## 11. References

- scikit-learn documentation  
- Census dataset description  