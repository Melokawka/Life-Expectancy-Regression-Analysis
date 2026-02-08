# Life-Expectancy-Regression-Analysis
The project is focused on predicting life expectancy based on socio-economic and health-related data (2,938 records, 21 features). The regression problem was transformed into a quasi-classification task by applying fuzzy logic and dividing the target variable into intervals. The data underwent preprocessing, including manual correction of corrupted columns (Population), missing data imputation (fuzzy class means, linear regression for correlated columns - e.g. HepatitisB), normalization, and feature selection (correlation analysis, VIF). The final model was a patternnet neural network with three hidden layers (24-12-6) - the best performance was achieved with the Levenberg–Marquardt method. The model reached an accuracy of approximately 0.898 and low regression errors (MAE/MAPE) even though the classes were heavily imbalanced.

# Conclusions

The application of fuzzification did not address the core nature of the problem - it remained a regression task, and converting it into classification did not provide significant benefits.

Imbalanced classes (especially the smallest class 1) negatively affected classification performance - the model tended to assign samples to more frequent classes.

Data preprocessing (imputation, normalization, manual correction of errors) was crucial for achieving good results, particularly given the large number of missing values and outliers.

The neural network achieved strong predictive performance, but regression metrics indicate that the classification interpretation was secondary and methodologically less justified.

Potential improvements include using pure regression instead of discretization, applying transformations to the target variable rather than interval-based classes, using PCA or other dimensionality reduction techniques, and implementing more advanced methods for handling class imbalance.
