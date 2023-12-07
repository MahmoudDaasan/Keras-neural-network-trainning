# Keras-neural-network-trainning
# Concrete Strength Prediction

This project involves predicting the strength of concrete using a neural network model. The code provided utilizes the Keras library for building and training the neural network on a concrete strength dataset.

## Overview

The main goal of this project is to demonstrate the implementation of a neural network for predicting concrete strength based on certain features. The code reads a concrete dataset, preprocesses the data, builds a neural network model, trains the model, and evaluates its performance.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Neural Network Model](#neural-network-model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Challenges](#challenges)


## Installation

Before running the code, ensure that the required dependencies are installed. You can install them using the following command:

```
pip install pandas scikit-learn keras numpy
```


## Dataset
The dataset that had been used can be called using the following command
 ```
 #Loading the dataset
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
```
## Neural Network Model
The neural network that had been created has 5 input nodes with Relu activation function, 1 hidden layer with 10 nodes with Relu activation function and finally the output layer with one node,
 ```
# Neural network model
def neural_network():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```
## Training 
The model is trained in the following manner:
 ```
# Training the model
for _ in range(num_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)
    model = neural_network()
    model.fit(X_train, y_train, validation_split=0.3, epochs=50, verbose=2)
```
The script uses a loop to perform multiple iterations of training, with data split into training and testing sets.
## Evaluation
Model evaluation is done as follows:
 ```
# Evaluating the model
score = model.evaluate(X_test, y_test, verbose=0)
```
The script calculates the mean squared error (MSE) between the predicted and actual concrete strength values.
## Results
The results are displayed as the mean and standard deviation of the MSE across multiple iterations:
 ```
# Displaying results
print("Mean squared error: " + str(mean))
print("Standard deviation: " + str(res))
```
