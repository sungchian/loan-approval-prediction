# 📌 Project Background 

Loan prediction is a real-world problem that many individuals face, especially for students who are going to graduate and are entering the workforce. They may need loans for various purposes, such as buying a car, securing housing, or furthering their education. Understanding the likelihood of loan approval is also crucial for them. Our report results could provide valuable insights and help individuals make informed decisions about their financial future.

Our project questions include:   
- Which features play a significant role in determining housing price classification?
- Which machine learning algorithms are best for our analysis?
- What business implications do our findings have? 

<i>In collaboration with Angel Sheu, Tracey Liu, and Lisa Rumao.</i>  

# ✨ Data Overview
- Source: [Dataset](https://www.kaggle.com/datasets/yashpaloswal/loan-prediction-with-3-problem-statement) from Kaggle.
- Size: There are 981 observations and 11 columns in the dataset.
- Demographic Features: Analyzed categorical columns like gender, marital status, dependent status, educational status, property area, credit history, and loan amount term.
- Numerical Features: Examined numerical columns such as applicant income, co-applicant income, and loan amount.


# 📊 Exploratory Data Analysis     
   <br>
      <img src="images/pic1.jpg" width="500">
   <br>  
-  
   <br>
      <img src="images/pic2.jpg" width="500">
   <br>  
- Property area is categorized as Urban, Semi-Urban, or Rural.
   <br>
      <img src="images/pic3.jpg" width="500">
   <br>  
- Credit history is a binary indicator (1: Good, 0: Bad)
   <br>
      <img src="images/pic4.jpg" width="500">
   <br>

These features contribute to a holistic understanding of applicants' backgrounds and financial capacities, laying the foundation for building an effective predictive model.

# 🔍 Benchmark 
We have carried out three machine-learning models: 
1. Decision Trees
2. Logistic Regression
3. Random Forest

Each model has its benchmark, which we will present in the analysis step.

The chart demonstrates the evaluation metrics for these three models.

The performance of each model is measured based on 
- Test Accuracy
- Test F1 Score
- Loan Acceptance Ratio
- Loan Rejection Ratio.
   <br>
      <img src="images/pic5.jpg" width="600">
   <br>  

# 🧽 Data Cleaning  (Todo)
### Logistic Regression
1. Change categorical variables into dummy variables & replace target variable with 0,1
   <br>
      <img src="images/pic6.jpg" width="400">
   <br>
2. Deleting the Missing Values
3. Standardized data
   <br>
      <img src="images/pic7.jpg" width="400">
   <br>
### Random Forest
1. Pre-pruning (mode): Implement pre-pruning techniques to avoid overfitting



# ✏️ Machine Learning Models  
### Logistic Regression

- We ran logistic regression with the cleaned data, which replaced NaN values with mode values and set the benchmarks 
   <br>
      <img src="images/pic8.jpg" width="400">
   <br>
   <br>
      <img src="images/pic9.jpg" width="400">
   <br>
- We also tried replacing categorical NaN values with mode values and numerical NaN values with the mean.
   <br>
      <img src="images/pic10.jpg" width="400">
   <br>
   <br>
      <img src="images/pic11.jpg" width="400">
   <br>
- Testing Different Parameters
  In the final process, we tested different parameters, such as the threshold, solver, and penalty, to find the best accuracy. To limit irrelevant coefficients, we chose Lasso Regularization to prevent overfitting and encourage sparsity in the model.
  <br>
      <img src="images/pic12.jpg" width="400">
  <br>
- We used GridSearchCV with CrossValidation to determine the best hyperparameters, and it turns out LogisticRegression(C=0.01, random_state=42, solver='liblinear') is the best parameter for the model. Eventually, we got the testing accuracy at 92%, and the F1 score was 95%
  <br>
      <img src="images/pic13.jpg" width="400">
  <br>

Our model assesses the accuracy threshold and F1 score for both the training and testing sets, producing a clear chart. After analyzing the decision threshold chart, we chose a threshold of 0.5 as it corresponds to the highest testing accuracy.
   <br>
      <img src="images/pic14.jpg" width="400">
  <br>

After fitting the logistic regression model with the L1 penalty, we inspect which variables have been set to zero (effectively deleted) by examining the coefficients of the model. 

Eventually, the model excluded all categorical variables and only kept numerical ones after implementing the L1 penalty.

In conclusion, the two methods that improved testing accuracy the most were removing missing values and applying L1 regularization. 

To determine whether to authorize the loans or not, we must still build a cost-benefit matrix and account for the costs associated with type 1 and type 2 errors because loan prediction is more closely tied to the expected value framework.

## 📈 Logistic regression accuracy chart:
  <br>
      <img src="images/pic15.jpg" width="400">
  <br>
   


# 🔑 Key Takeaways    
The performance of the Logistic Regression model remained steady across various methods, showcasing notable advancements, particularly when leveraging GridSearchCV for parameter optimization.

While the Random Forest model demonstrated enhanced accuracy through techniques like pre-pruning and parameter testing, the impact of the 'Delete missing values' method was moderate.

In contrast, the Decision Tree model displayed consistent performance regardless of the method employed.

In conclusion, for maximizing model performance, Logistic Regression exhibited superior results, especially when fine-tuning parameters, closely trailed by the Random Forest model. The Decision Tree model did not substantially improve accuracy following the 'Delete missing values' method.

# ☁️ Project Improvements  






 
