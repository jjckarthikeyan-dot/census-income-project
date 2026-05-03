# Census Income Classification & Customer Segmentation

## Overview

In this project, I worked with U.S. Census data to solve a practical business problem for a retail client.

The main idea was simple:

* Can we predict whether someone earns more than $50K?
* Can we group customers into meaningful segments for marketing?

The dataset includes demographic and employment-related details such as age, education, occupation, and work status. It also includes a **weight column**, since the data comes from a sampled population.

---

## Project Structure

```
census_income_project/
  data/
    census-bureau.data
    census-bureau.columns
  src/
    train_classifier.py
    segmentation.py
  outputs/
  requirements.txt
  REPORT.md
  README.md
```

---

## Setup

First, create a virtual environment:

python -m venv .venv

Activate it:

**Windows (CMD):**


.venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

---

## Data Setup

Create a folder named `data` in the project root and place the two files inside it:

data/census-bureau.data
data/census-bureau.columns

---

## Running the Classification Model

python src/train_classifier.py

This script trains a model that predicts whether a person belongs to the `<=50K` or `>50K` income group.

### Outputs

outputs/income_classifier.joblib
outputs/classification_metrics.json
outputs/feature_importance.csv

### Notes

* I used the `weight` column during training and evaluation because the dataset represents a sampled population.
* Instead of focusing only on accuracy, I used metrics like:

  * ROC-AUC
  * Precision-Recall (PR-AUC)
  * Balanced Accuracy

This is important because the `>50K` group is smaller, and accuracy alone can be misleading.

---

## Running the Segmentation Model
python src/segmentation.py


This script groups customers into segments based on demographic and employment-related features.

### Outputs
outputs/segment_profile.csv
outputs/segmentation_summary.json
outputs/census_with_segments_sample.csv

---

## Approach

### 1. Classification
For the classification task, I built a model that learns patterns from features such as:

* age
* education
* occupation
* work status

and predicts whether someone is likely to earn more than $50K.

I treated this as a real-world problem, so I considered:

* class imbalance
* weighted data
* business relevance of predictions

---

### 2. Segmentation

For segmentation, I did something different.

I **did not use the income label during training**.
Instead, I grouped people based on their characteristics.

After creating the segments, I analyzed:

* how income varies across segments
* what defines each group

This makes the segmentation more useful for real marketing use cases.

---
## Business Value

From a business perspective, this project can help the client:

* identify high-income customers for premium targeting
* understand budget-sensitive groups
* design personalized campaigns
* improve customer engagement strategies

Each segment behaves differently, so marketing strategies can be tailored accordingly.

---
## Key Files

| File                                  | Description                                  |
| ------------------------------------- | -------------------------------------------- |
| `src/train_classifier.py`             | trains and evaluates income prediction model |
| `src/segmentation.py`                 | builds customer segments                     |
| `outputs/classification_metrics.json` | model performance                            |
| `outputs/feature_importance.csv`      | important features                           |
| `outputs/segment_profile.csv`         | segment summary                              |
| `REPORT.md`                           | detailed project explanation                 |

---

## Final Note
I focused on building a solution that is not just technically correct, but also useful from a business perspective.

The goal was to understand the data, make reasonable modeling decisions, and explain the results in a way that a non-technical stakeholder can understand.
