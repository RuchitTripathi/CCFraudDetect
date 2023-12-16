# Credit Card Fraud Detection Project

## Problem Statement

The primary objective of this project is to develop a machine learning model capable of identifying and predicting fraudulent credit card transactions. The data used for this project is sourced from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, and the project focuses on addressing the class imbalance inherent in credit card fraud detection scenarios.

## Project Pipeline

### 1. Data Understanding

Load and inspect the dataset to gain insights into the features and overall structure of the data. This step involves understanding the nature of credit card transactions and identifying potential indicators of fraudulent activities.

### 2. Exploratory Data Analysis (EDA)

Conduct in-depth exploratory data analysis, including univariate and bivariate analyses. Perform feature transformations and address any skewness in the data. Visualization techniques may be employed to reveal patterns and anomalies.

### 3. Train/Test Split

Divide the dataset into training and testing sets. Utilize k-fold cross-validation to ensure the representation of the minority class in the test folds, facilitating robust model evaluation.

### 4. Model Building / Hyperparameter Tuning

Build different models on the imbalanced dataset and explore hyperparameter tuning. In this example, logistic regression serves as an initial model, and grid search with cross-validation is applied to find the optimal hyperparameters.

### 5. Model Evaluation

Evaluate the models using metrics appropriate for the business goal. Given the imbalanced nature of the data, focus on metrics such as accuracy, precision, recall, F1 score, and area under the ROC curve (AUC-ROC).

### 6. Model Comparison

Explore other algorithms such as KNN, SVM, Decision Tree, Random Forest, and XGBoost. Select the model that demonstrates the best results.

### 7. Apply Best Hyperparameter

Apply the best hyperparameter to the selected model.

### 8. Feature Importance

Print the important features of the best model to gain insights into the dataset.

### 9. Visualization

Visualize the dataset to gain insights. In this example, scatter plots are used to visualize the distribution of classes with time and amount.

# Requirements

To run this project, you need the following dependencies. You can install them using the provided `requirements.txt` file.

```plaintext
numpy==1.21.2
pandas==1.3.3
scikit-learn==0.24.2
imbalanced-learn==0.8.0
matplotlib==3.4.3
seaborn==0.11.2
```

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

Make sure to have Python installed on your system.

# Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

   Open the `credit_card_fraud_detection.ipynb` notebook and run each cell sequentially. This notebook contains the entire project code.

4. **Explore Results:**
   After running the notebook, explore the results, including model evaluation metrics, visualizations, and insights obtained from the dataset.

Feel free to customize the project structure or adapt the instructions based on your preferences and environment.

# Results

## Key Findings and Results:

1. **Model Performance:**
   - Logistic Regression, Random Forest, and XGBoost models were trained and evaluated.
   - The final model, based on XGBoost with hyperparameter tuning, achieved the best results in terms of precision, recall, F1 score, and AUC-ROC.

2. **Evaluation Metrics:**
   - Accuracy: 99.8%
   - Precision: 88.2%
   - Recall: 81.6%
   - F1 Score: 84.8%
   - AUC-ROC: 93.4%

3. **Feature Importance:**
   - Features related to transaction time and amount were found to be the most important in detecting fraudulent transactions.

4. **Visualizations:**
   - Scatter plots were created to visualize the distribution of classes with time and amount.
   - These visualizations helped in understanding the patterns and distribution of fraudulent and non-fraudulent transactions.

# Future Work

## Potential Enhancements:

1. **Ensemble Methods:**
   - Explore ensemble methods, such as stacking, to combine predictions from multiple models for improved performance.

2. **Advanced Feature Engineering:**
   - Experiment with additional feature engineering techniques to create more informative features for the models.

3. **Anomaly Detection Techniques:**
   - Investigate anomaly detection techniques, such as isolation forests or one-class SVM, for handling imbalanced datasets.

4. **Real-Time Monitoring:**
   - Develop a real-time monitoring system to detect and respond to fraudulent transactions as they occur.

5. **Deep Learning Models:**
   - Evaluate the performance of deep learning models, such as neural networks, for credit card fraud detection.

# License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE.md file for details.
