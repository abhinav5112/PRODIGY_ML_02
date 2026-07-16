# Customer Segmentation using K-Means Clustering

## Overview

This project implements the K-Means Clustering algorithm to segment retail store customers based on their Annual Income and Spending Score. Customer segmentation helps businesses understand different customer groups and create targeted marketing strategies.

This project was completed as **Task-02** of the **Prodigy InfoTech Machine Learning Internship**.

---

## Objective

The objective of this project is to:

- Perform customer segmentation using K-Means Clustering.
- Find the optimal number of clusters using the Elbow Method.
- Visualize customer groups.
- Save the clustered dataset for further analysis.

---

## Dataset

**Mall Customer Segmentation Dataset**

Dataset Columns:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

Dataset Source:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Algorithm

- K-Means Clustering (Unsupervised Learning)

---

## Steps Performed

1. Import required libraries.
2. Load the customer dataset.
3. Explore and analyze the dataset.
4. Select Annual Income and Spending Score as features.
5. Apply the Elbow Method to determine the optimal number of clusters.
6. Train the K-Means model.
7. Visualize customer clusters.
8. Save the clustered dataset.

---

## Output

The project generates:

- Elbow Method Graph
- Customer Segmentation Scatter Plot
- Customer_Segmentation_Result.csv

---

## Project Structure

```
Task-02/
│
├── customer_segmentation.py
├── customer_segmentation.ipynb
├── Mall_Customers.csv
├── Customer_Segmentation_Result.csv
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Use additional customer features for segmentation.
- Apply Hierarchical Clustering.
- Create an interactive dashboard using Streamlit.
- Deploy the model as a web application.

---

## Author

**Abhinav Kumar**

B.Tech Computer Science Engineering (Blockchain Technology)

Machine Learning Intern at Prodigy InfoTech

---

## Acknowledgement

This project was completed as part of the **Prodigy InfoTech Machine Learning Internship Program**.
