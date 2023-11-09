# deep-learning-challenge

![Thin--1--7](https://github.com/dspataru/deep-learning-challenge/assets/61765352/219f0078-fe95-4ef0-8c6d-eeb51703b7aa)

## Table of Contents
* [Background]()
* [Overview]()
* [Data Source]()
* [Methodology]()
* [Results]()
* [Summary]()

## Background

In today's data-driven world, the application of artificial intelligence and machine learning has become instrumental in addressing complex challenges, and one such challenge is to predict the success of charity funding applicants. Neural networks, a subset of machine learning, have emerged as a powerful tool in solving binary classification problems, which involve categorizing data into two distinct groups or outcomes. In this report, we will explore the development, optimization, and evaluation of a neural network model for a charity organization using the "charity_data.csv" dataset.

### Background on Neural Networks
Neural networks are computational models inspired by the human brain's structure and functioning. They consist of interconnected nodes (neurons) organized into layers, namely the input layer, hidden layers, and the output layer. Neural networks excel in their ability to learn intricate patterns and relationships within data, making them well-suited for a wide range of applications, including binary classification.

![0_IlHu39jf2c7QC4kn](https://github.com/dspataru/deep-learning-challenge/assets/61765352/cb79b1cf-b256-4de8-99c6-951e78e0fa07)

Neural networks have gained prominence in recent years due to their capacity to handle complex, high-dimensional datasets. They are particularly adept at discerning patterns in data that may be challenging for traditional statistical models to capture. This makes them an ideal choice for addressing problems like predicting the success of charity funding applicants, where a multitude of factors can influence the outcome.

### Uses of Neural Networks in Binary Classification
Binary classification is a common problem in machine learning, where the goal is to categorize data into one of two classes or outcomes. Neural networks have proven to be highly effective in this context due to their ability to model complex relationships, identify nonlinear patterns, and adapt to different types of data.

In the realm of charity funding, binary classification is often applied to predict whether an applicant will be successful if funded. Neural networks can analyze a diverse range of features, such as applicant demographics, financial information, and project details, to make informed predictions. By learning from historical data, neural networks can assist charity organizations in making objective decisions about allocating their resources.

#### Key Words
deep learning model, neural networks, optimization, jupyter notebook, google colab, pythons, binary classification, pandas, tensorflow, keras_tuner, sklearn, train and test split, standardscalar, machine learning models, funding prediction, hyperparameter tuning, hidden layers, activiation function

## Overview

This report will detail the steps involved in developing a neural network model for binary classification, from data preprocessing to model optimization. The primary objective is to enhance the charity organization's ability to identify potential beneficiaries effectively, thereby maximizing the impact of their charitable efforts.

## Data Source

The data is provided by Alphabet Soup's business team in the form of a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively?

Below is a snapshot of the first few columns and rows of the dataset:

![dataset](https://github.com/dspataru/deep-learning-challenge/assets/61765352/20ae8d79-0b69-4787-8f51-33951007c607)


## Methodology

### Preprocess the Data

Pandas and scikit-learn's `StandardScaler()` function was used to preprocess the data before compiling, training, and evaluating the neural network model. 

1. **Target and Feature Variables**: The first step is to create read the "charity_data.csv" file and identify the target and feature variables in the dataset. The target variable for our model is "IS_SUCCESSFUL," which indicates whether the funding provided was used effectively (1 for success, 0 for failure), and the feature variables include metadata about the organizations, such as "APPLICATION_TYPE," "AFFILIATION," "CLASSIFICATION," "USE_CASE," "ORGANIZATION," "STATUS," "INCOME_AMT," "SPECIAL_CONSIDERATIONS," and "ASK_AMT."
2. **Data Cleanup**: We dropped the "EIN" and "NAME" columns as they are identification columns and do not provide valuable information for the model.
3. **Unique Values**: We determined the number of unique values in each column to understand the diversity of data.
4. **Binning Rare Categories**: For columns with more than 10 unique values, we binned rare categorical variables into a new category called "Other." This helps prevent overfitting and simplifies the model.
5. **Encoding Categorical Variables**: We used pd.get_dummies() to encode categorical variables, converting them into numerical form for model input.
6. **Splitting Data**: The data was split into training and testing datasets to train and evaluate the model's performance.
7. **Scaling Data**: The data was scaled using a StandardScaler fitted to the training data, ensuring that all features have a similar scale for model convergence.

Below is a snippet of the data after pre-processing:

![data_cleaned](https://github.com/dspataru/deep-learning-challenge/assets/61765352/b6311231-c966-4747-87db-e0841615fab1)


### Compile, Train, and Evaluate the Model
1. **Neural Network Model**: We designed a neural network model with input features and nodes in each layer using TensorFlow and Keras.
2. **Model Architecture**: The neural network model consists of an input layer, one hidden layer with ReLU activation, and an output layer with a sigmoid activation function.
5. **Training and Evaluation**: The model was compiled and trained with a callback to save weights every five epochs.
We evaluated the model's performance using the test data, measuring loss and accuracy.
6. **Model Export**: The trained model results were saved and exported to an HDF5 file named "AlphabetSoupCharity.h5."

The initial model was defined by the following code:
```python

# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=80, activation="tanh", input_dim=num_cols))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=30, activation="tanh"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

```

### Optimize the Model
1. **Repeated Preprocessing**: We repeat the preprocessing steps in a new notebook to ensure consistency.
2. **Create New Model**: We create a new neural network model implementing at least three model optimization methods to improve model performance.
3. **Save and Export**: The results of the optimized model are saved and exported to an HDF5 file named "AlphabetSoupCharity_Optimization.h5".


## Results and Analysis

The initial model comprised of two hidden layers, input dimension = 43, and used the tanh function as the activation function. The number of neurons in the first hidden layer is 80, and the second hidden layer is 30. After evaluating the model with the test data, the accuracy of the model was calculated to be 0.7258 or 72.58%. The loss was calculated to be 0.5610.

Three different methods were used in an attempt to increase the accuracy of the model:
1. Changing Input Data & Reducing the Cutoff Value of Applications:
  * Dropping CLASSIFICATION and SPECIAL_CONSIDERATIONS columns in addition to EIN and NAME.
  * Cutoff value for apps is 10.
  * Activation function: relu.
  * NUmber of hidden layers: 2.
2. Reducing the Cutoff Value for Applications and Classification, and Increasing Hidden Layers.
  * Dropping only the EIN and NAME columns.
  * Cutoff value for apps is 10.
  * Cutoff value for class is 70.
  * Activation function: tanh.
  * Number of hidden layers: 3.
3. Hyperparameter Tuning
  * A function is written to create a model with different parameter values.
  * The function allows kerastuner to decide which activation function to use in the hidden layers, the number of neurons in the first layer, and the number of hidden layers and neurons in hidden layers.
  * Constraints for the different input parameters:
      * Activation functions: relu & tanh.
      * Number of neurons for the first layer: between 1 and 30, incrementing by 5 each iteration.
      * Number of hidden layers: between 1 and 5.
      * Number of neurons in each hidden layer: between 1 and 30, incrementing by 5 each iteration.
  * Here we use the same input data as the original model uses.

#### Summary of Results

![results_summary](https://github.com/dspataru/deep-learning-challenge/assets/61765352/993f02a8-9c21-4a62-9dac-a54a5ccfd82a)

The target model performance of 75% was not achieved with the different optimization method attempts.

## Conclusion

The highest accuracy the deep learning model was able to achieve based on the dataset provided was 72.77%. Several optimization methods were explored in an attempt to increase the accuracy of the model. Several observations were made:
1. Reducing the number of features had a negative effect on the performance of the model.
2. Hyperparameter optimization within the bounds that were set for each parameter was not successful in finding a set of parameters for the model that improved the accuracy from the original model.
3. Increasing the number of hidden layers did not improve the accuracy of the model.
4. Changing the activation function did not have a significant impact on the accuracy of the model.

It is possible that if the search space for the hyperparamitization method was increased, we may have been able to find more optimal parameters for the model to achieve the target accuracy of 75%. There are other models that could have been explored outside of neural networks, which include, decision trees, random forests, K-nearest neighbors, support vector machines, and ensemble methods. 

The choice of the appropriate model depends on factors like dataset size, feature types, computational resources, interpretability, and the specific requirements of the binary classification problem. In the future, it would be good to explore the different models and compare their performance to the neural network models explored in this project.
