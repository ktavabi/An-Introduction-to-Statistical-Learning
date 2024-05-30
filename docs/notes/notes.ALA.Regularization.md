---
id: 47fl40zr5gbh9tk7c7jtezh
title: Regularization
desc: "Coursera Advanced Learning Algorithms regularization bias/variance"
updated: 1716446135089
created: 1716446084601
---

# Regularization and bias/variance

The choice of the regularization parameter Lambda affects the bias and variance of a learning algorithm. A large value of Lambda leads to high bias and underfitting, while a small value of Lambda leads to high variance and overfitting. Cross-validation is a way to choose a good value of Lambda. Different values of Lambda and performance evaluation on a cross-validation set can help to identify the best value for the regularization parameter. The training error and cross-validation error vary as a function of Lambda.

The choice of the regularization parameter Lambda directly affects the bias and variance of a learning algorithm. Here's how:

1. Large Lambda (high regularization):

   - Bias: When Lambda is set to a large value, the algorithm is motivated to keep the parameters small. This leads to a simpler model with high bias, as it underfits the training data.
   - Variance: With high regularization, the algorithm tends to have smaller parameter values, resulting in a model that is less flexible. This reduces the variance but may lead to underfitting.

2. Small Lambda (low regularization):

   - Bias: When Lambda is set to a small value or even zero, there is no regularization, and the algorithm fits the training data more closely. This can result in a model with low bias, as it can capture more complex patterns in the data.
   - Variance: With low or no regularization, the algorithm has more freedom to fit the training data precisely. This can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data, resulting in high variance.

3. Intermediate Lambda:

   - By choosing an intermediate value of Lambda, you can strike a balance between bias and variance. This value allows the algorithm to fit the training data well while still generalizing to new data. It helps in achieving a good trade-off between underfitting and overfitting.

In summary, the choice of the regularization parameter Lambda allows you to control the bias and variance of a learning algorithm. By selecting an appropriate value, you can find a model that fits the data well without overfitting or underfitting.

Cross-validation can help in choosing the optimal value of the regularization parameter Lambda by evaluating the performance of the learning algorithm for different values of Lambda on a separate validation set. Here's how it works:

1. Define a range of Lambda values: Start by defining a range of possible values for Lambda that you want to explore. This range should cover a wide spectrum, including both small and large values.
2. Train the model for each Lambda value: For each value of Lambda in the defined range, train the learning algorithm using that Lambda value. This involves minimizing the cost function and obtaining the corresponding model parameters.
3. Evaluate performance on the validation set: After training the model for each Lambda value, evaluate the performance of the model on a validation set using a performance metric such as the cross-validation error. This metric provides an estimate of how well the model generalizes to unseen data.
4. Choose the optimal Lambda value: Compare the performance of the model for different Lambda values on the validation set. Look for the Lambda value that results in the lowest cross-validation error. This Lambda value is considered the optimal choice as it indicates the best trade-off between bias and variance.
5. Test the model: Finally, to further validate the chosen Lambda value, test the model's performance on a separate test set. This provides an estimate of the model's generalization error.

By using cross-validation, you can systematically evaluate the performance of the learning algorithm for different values of Lambda and select the one that yields the best performance on unseen data. This helps in choosing the optimal value of Lambda for regularization, ensuring that the model achieves the right balance between bias and variance.
