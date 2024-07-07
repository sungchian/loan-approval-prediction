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
      <img src="Images/img-01.png" width="500">
   <br>  
-  
   <br>
      <img src="Images/img-02.png" width="500">
   <br>  
- Property area is categorized as Urban, Semi-Urban, or Rural.
   <br>
      <img src="Images/img-03.png" width="500">
   <br>  
- Credit history is a binary indicator (1: Good, 0: Bad)
   <br>
      <img src="Images/img-04.png" width="500">
   <br>
- 
   <br>
      <img src="Images/img-05.png" width="500">
   <br>

These features contribute to a holistic understanding of applicants' backgrounds and financial capacities, laying the foundation for building an effective predictive model.

# 🔍 Benchmark 
We have carried out three machine-learning models: 
1. Decision Trees
2. Logistic Regression
3. Random Forest

Each model has its own benchmark, which we will present in the analysis step.

The chart demonstrates the evaluation metrics for these three models.

The performance of each model is measured based on 
- Test Accuracy
- Test F1 Score
- Loan Acceptance Ratio
- Loan Rejection Ratio.


# 👣 Our Approach  
Real estate analysis typically attempts to predict price, a continuous variable. However, we took a classification approach to this problem, since classification introduces a layer of interpretability and simplicity to our analysis, which can benefit business professionals and prospective buyers. By categorizing properties into pricing tiers (high and low), we aim to compare the accuracy and performance of each of the selected models.    

# 🧽 Data Cleaning  
1. We discretized the dependent variable.   
   <br>
      <img src="Images/img-06.png" width="600">
   <br>  
2. Then, we filled in the missing values with the mean and median values. Specifically, we used the median for the missing values in the acre_lot, house_size, and price columns. Additionally, rows with missing values in the city and zip_code columns were removed. Lastly, the records for Tennessee, South Carolina, and Virginia listings were removed because they contained a substantial amount of missing data.  
   <br>
      <img src="Images/img-07.png" width="600">
   <br>

This "cleaned" dataset served as our initial benchmark for subsequent machine learning experiments.  

# 🔍 Machine Learning Models  
### Random Forest Classifier    
- Without any pre-processing techniques, the results are as follows:  
   <br>
      <img src="Images/img-08.png" width="400">
   <br>  
The model predicts approximately 92% of instances correctly. The precision and recall of the model are relatively high, at 0.92, indicating a low rate of false positives and negatives. Additionally, the high F1-score implies that this model performs well. Overall, the initial benchmark for the Random Forest algorithm on this dataset demonstrates strong performance with high accuracy.  
- We attempted to enhance the model with random sampling, dummy variables for the state attribute, feature selection, binning, min-max scaling, and standardization pre-processing techniques. Random sampling reduced the dataset for additional features while maintaining a similar model accuracy. Implementing dummy variables for the state attribute did not change the accuracy. Since the number of attributes in the original dataset is not extremely large, feature selection was not useful. Binning underscored the nature of the data which decreased its accuracy. Standardization improved the previous pre-processing, but it had minimal effect on improving the accuracy.  
   <br>
      <img src="Images/img-09.png" width="400">
   <br>  
Overall, the Random Forest model that performed the best was the benchmark model (with no pre-processing). Many of the additional pre-processing techniques either worsened or had no impact relative to the original accuracy. However, a finding that was gained from the pre-processing was that price is likely to be influenced by location since adding the dummy variables for the state attribute improved the randomly sampled model.

### K-Nearest Neighbors Classifier  
- Without any pre-processing techniques, the results are as follows:  
   <br>
      <img src="Images/img-10.png" width="400">
   <br>  
This model predicts approximately 88% of instances correctly. The model's precision and recall are commendably high, at 0.87, suggesting a low frequency of false positives and negatives. The high F1-score further confirms the model's effectiveness. However, this model appears to be less accurate than the Random Forest model with a 92% overall accuracy.
- The pre-processing techniques that we employed were random sampling, an introduction of dunny variables for the states attribute, feature selection, binning, min-max scaling, and standardization. Most of the techniques were moderately effective, but they tended to distort the true nature of the data, leading to less accurate predictions. Standardization offered some improvement over the other pre-processing techniques. However, the impact it had on the overall accuracy was insignificant.     
   <br>
      <img src="Images/img-11.png" width="400">
   <br>  
Overall, the KNN model achieved optimal results when applied to the standardization of the bed, bath, acre_lot, and house_size attributes. The standardized KNN model achieved an 88.5% accuracy. a slight increase of 0.1% from the benchmark model.  

### Logistic Regression  
- Without any pre-processing techniques, the results are as follows:  
   <br>
      <img src="Images/img-12.png" width="400">
   <br>  
This model predicts approximately 70% of instances correctly. The model's precision and recall contained mixed results. As noted by the F1-score of 0.65 for the "high" class and 0.74 for the "low" class, this seems to suggest that the model is better at identifying the "low" class data points. Overall, this Logistic Regression model has room for improvement, particularly in capturing the "high" class data points.
- Similar to the Random Forest and KNN models, the pre-processing techniques that we employed were random sampling, an introduction of dunny variables for the states attribute, feature selection, binning, min-max scaling, and standardization. Implementing dummy variables for the states attribute increased the model's accuracy to 72%. Standardization and binning on the house_size attribute improved the model's accuracy to 75%.  
   <br>
      <img src="Images/img-13.png" width="400">
   <br>  
Overall, the Logistic Regression approach had better performance in two separate instances, binning on the house_size attribute and a combination of random sampling, dummy variables, and standardization. Both models achieved an accuracy of around 75%, an improvement of approximately 5% from the benchmark model.

# 🔑 Key Takeaways    
We implemented three machine learning models to predict whether a real estate listing could be classified as a "high" or "low" price listing. From our analysis, we concluded how different models reacted to a variety of pre-processing techniques. From our choice of pre-processing techniques, the best techniques seemed to involve a combination of random sampling, standardization, and dummy variables for the state attribute. The binning technique improved the Logistic Regression model significantly. However, binning also led to a significant decrease in the performance of the Random Forest and KNN models, suggesting the importance of retaining the original granularity for some features.  

For this dataset, the best model seemed to be the Random Forest algorithm with no pre-processing. This model had the highest accuracy of 92%. Property location also seemed to be the attribute that plays a significant role in price.  

# ☁️ Project Improvements  
This project was for the first machine learning class that I took, and it was also the first project where I applied machine learning algorithms. Knowing what I know now if I were to improve this project, I would focus on improving the Random Forest model using different boosting methods, such as Adaptive Boosting and Gradient Boosting.  





 
