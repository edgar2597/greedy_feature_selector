## Data descritption 

I am using Wisconsin Breast Cancer dataset (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). 
I am using ‘diagnosis’ as my target variable Y to be predicted. Also I replace Malignant by 1, and Benign by 0, thus predicting cancer malignancy


## Algorithm

I implement simplest best model selection as a fast-forward “greedy” algorithm, which works like this:
-	Finding the best one-feature model by trying all one-feature models, and selecting the one with the lowest error. This is our best feature F1.
-	Using F1 from the first step,  adding one more feature to it from all features left, to find the best 2-feature model (F1, F2)
-	Similarly, adding more features: F3, F4, F5 – to the features from the previous step

For solving classification problem, I use LogisticRegression model.
