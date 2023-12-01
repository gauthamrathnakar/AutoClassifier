# AutoClassifier
This project automatically selects the best classification algorithm for a pre-processed dataset. It implements 6 popular algorithms and evaluates their accuracy on a testing set. The algorithm with highest accuracy is chosen as the best for the dataset.

## Project Description

This project focuses on the automatic selection of the best classification algorithm for a given pre-processed dataset. It implements and evaluates six popular algorithms:

1. **Logistic Regression:** A linear model for binary classification.
2. **K-Nearest Neighbors (KNN):** Classifies data points based on the majority class of their closest neighbors.
3. **Support Vector Machines (SVM):** Creates a hyperplane in the high-dimensional space to maximize the margin between classes.
4. **Decision Tree:** A tree-based model that makes decisions based on a series of rules learned from the data.
5. **Random Forest:** An ensemble of decision trees that improves accuracy through voting.
6. **Naive Bayes:** A probabilistic model that assumes features are independent given the class.

## Functionality

The project takes a pre-processed dataset as input and performs the following steps:

1. **Splits the data into training and testing sets:** This ensures the model is not trained on data it will be evaluated on.
2. **Trains each of the six algorithms on the training set:** Each algorithm learns the patterns in the data.
3. **Evaluates the accuracy of each algorithm on the testing set:** This measures how well each algorithm generalizes to unseen data.
4. **Selects the algorithm with the highest accuracy:** This is considered the best algorithm for the specific dataset.

## Outputs

The project outputs the following:

* The accuracy of each algorithm on the testing set.
* The algorithm with the highest accuracy.

## Use Cases

This project can be used by researchers and practitioners to:

* **Automate the selection of the best classification algorithm for a given dataset.**
* **Benchmark the performance of different algorithms on a specific task.**

## Technologies

The project is implemented using Python and popular machine learning libraries such as scikit-learn, pandas and numpy

## Next Steps

The project can be further improved by:

* Implementing additional classification algorithms.
* Integrating feature selection techniques.
* Exploring hyperparameter tuning for further performance optimization.
* Providing detailed documentation and examples for easier use.

## Contributing

Contributions to the project are welcome! Please see the contributors' guide: CONTRIBUTING.md for details.

