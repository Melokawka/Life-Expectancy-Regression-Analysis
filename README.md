# Life-Expectancy-Regression-Analysis
The project is focused on predicting life expectancy based on socio-economic and health-related data (2,938 records, 21 features). 

The project was used to test the idea of transforming a typical regression problem into a classification task using fuzzification of the Life Expectancy column. As opposed to naive approaches, fuzzy classes preserve the entirety of the original data structure due to reversibility of the fuzzification.

The data underwent preprocessing, including manual correction of corrupted columns (Population), missing data imputation (fuzzy class means, linear regression for correlated columns - e.g. HepatitisB), normalization, and feature selection (correlation analysis, VIF). The final model was a patternnet neural network with three hidden layers (24-12-6) - the best performance was achieved with the Levenberg–Marquardt method. The model reached an accuracy of 0.898 even though the classes are heavily imbalanced.

# Conclusions

Fuzzification of the Life Expectancy parameter did not simplify the problem - it remained a regression task, and converting it into classification did not provide significant benefits.

Imbalanced classes (especially the smallest class 1 that included countries with the life expectancy of 35-45) negatively affected classification performance - the model tended to assign samples to more frequent classes.

Data preprocessing (imputation, normalization, manual correction of errors) was necessary for achieving good results, particularly given the large number of missing values and outliers.

The neural network achieved acceptable accuracy in classifying countries to their life expectancy interval.

The model can be further improved by: 
1. Using pure regression instead of transforming the problem to a classification task,
2. Using PCA or other dimensionality reduction techniques,
3. Implementing more advanced methods for handling class imbalance.
