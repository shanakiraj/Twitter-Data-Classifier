# Twitter-Data-Classifier

## Project Description

The goal of this project is to build machine learning models that can predict bias in Twitter data. The project involves preprocessing the text data, vectorizing the data, splitting the data into training and testing sets, training different classification models, and evaluating the performance of these models. For anyone who wants to take a look and run the code without downloading locally use this link: https://www.kaggle.com/shanakiraj/twitterclassifier

### Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Data

The dataset used in this project is taken from Kaggle and includes 80,000 tweets labeled as either "biased" or "not biased." The tweets were collected using various search terms related to politics and news events.

### Preprocessing

Before vectorizing the data, the text data was preprocessed to remove URLs, mentions, special characters, and stop words. The text data was then converted to lowercase and tokenized.

### Vectorization

The preprocessed text data was then vectorized using a CountVectorizer from Scikit-learn. The vectorized data was then split into training and testing sets.

### Model Training

Two classification models were trained on the training set:

* Support Vector Machine (SVM)
* Naive Bayes

The hyperparameters for each model were tuned using grid search and cross-validation. The performance of each model was evaluated using accuracy, precision, recall, and F1-score.

## Results

The results of the model training and evaluation are as follows:

### Support Vector Machine (SVM)
Hyperparameters tested:

* Kernel: linear, polynomial, RBF
* C: 0.1, 1, 10, 100
Best hyperparameters:

* Kernel: linear
* C: 0.1

Validation accuracy with best hyperparameters: 0.7654

Test accuracy with best hyperparameters: 0.7752

Precision with best hyperparameters: 0.7695

Recall with best hyperparameters: 0.9842

F1-score with best hyperparameters: 0.8637

### Naive Bayes

Hyperparameters tested:

* Alpha: 0.1, 1, 10, 100
* Fit_prior: True, False

Best hyperparameters:

* Alpha: 1
* Fit_prior: True

Validation accuracy with best hyperparameters: 0.7445

Test accuracy with best hyperparameters: 0.7597

Precision with best hyperparameters: 0.8616

Recall with best hyperparameters: 0.8026

F1-score with best hyperparameters: 0.8311

### Conclusion

Overall, the SVM model performed the best in terms of accuracy, precision, recall, and F1-score. However, the Naive Bayes model also performed well, especially in terms of precision. The Logistic Regression model performed the worst
