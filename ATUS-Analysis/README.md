# **Predicting Time Spent at Home: Insights from the American Time Use Survey**

#### **Objective:**  
This project focuses on predicting the amount of time individuals spend at home daily based on various socio-economic and demographic factors. Using the American Time Use Survey (ATUS) data (2010-2019), the goal was to engineer meaningful features, preprocess the data effectively, and build robust predictive models.

---

#### **Key Steps:**

1. **Data Collection and Cleaning:**  
   - Integrated multiple ATUS datasets, ensuring consistency across the years.
   - Filtered for primary respondents and prepared data for time-at-home analysis.
   - Addressed data quality issues such as duplicate cycles and invalid activity durations.

2. **Feature Engineering:**  
   - Created condensed features for demographics like race, household size, and income to handle imbalances in categorical data.  
   - Generated new attributes, such as "Num_Adults" and "Num_Children," to provide deeper insights into household composition.  
   - Leveraged meaningful transformations like identifying weekdays versus weekends.

3. **Model Building and Evaluation:**  
   - Compared three advanced models: LightGBM, XGBoost, and CatBoost.  
   - Conducted hyperparameter tuning for each model using Optuna for optimal performance.  
   - Evaluated models using RMSE to determine predictive accuracy and reliability.  

---

#### **Results:**  
- **Model Performance:**  
   - LightGBM achieved an RMSE of **3.371** (±0.199), making it the most balanced model in terms of accuracy and simplicity.  
   - XGBoost and CatBoost also performed competitively, with RMSEs of **3.355** and **3.360**, respectively.  

- **Feature Insights:**  
   - Household composition (number of adults and children) strongly influenced time spent at home.  
   - Condensed income levels and education categories provided significant predictive power while maintaining simplicity.  

---

#### **Conclusion:**  

This project successfully demonstrated the ability to predict the time individuals spend at home using socio-economic factors from the ATUS dataset. The best-performing model, LightGBM, achieved an RMSE of **3.371**, which translates to an error of approximately 3.371 hours in a 24-hour day. While this is a significant accomplishment, it’s important to contextualize the results within the broader scope of the data and the problem:

1. **Comparison to Data Variability**:  
   The standard deviation of time spent at home in the dataset was approximately **4.7 hours**, indicating substantial variability in the target variable. The model's performance, therefore, suggests it captures much of the systematic behavior in the data but cannot fully account for this inherent variability.

2. **Limitations of the Data**:  
   The quality of the ATUS data, while comprehensive, may include inaccuracies, reporting biases, or missing contextual factors that limit the model's ability to generalize. Factors like subjective interpretation of "time at home" and unrecorded variables (e.g., sudden life events, nuanced behavior patterns) could contribute to prediction errors.

3. **Inherent Uncertainty in Human Behavior**:  
   Human life is inherently random and influenced by countless unpredictable factors (e.g., spontaneous decisions, emergencies). This randomness inherently limits the precision of any predictive model in this context.

#### Final Remarks
Despite these limitations, the project highlights the potential of machine learning to analyze and predict human behavior at scale, offering valuable insights into socio-economic patterns. Future work could focus on enhancing data quality, incorporating additional contextual variables, or using ensemble approaches to further reduce error margins. However, it’s important to recognize the natural bounds of predictability for such complex and dynamic systems.

#### **Future Work:**  
- Incorporate post-COVID-19 datasets to analyze behavioral shifts.
- Explore additional models like neural networks for enhanced prediction capabilities.
- Develop an interactive dashboard for visualizing time-use patterns across demographics. 

---
