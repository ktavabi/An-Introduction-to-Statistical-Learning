---
id: u7cf2kcxzbprf5e7wj1xyb9
title: Model Performance
desc: Coursera Advanced Learning Algorithms establishing baseline model performance.
updated: 1716446029283
created: 1716445953636
---

# Establishing a baseline level of performance

The training error and cross-validation error are two metrics used to evaluate the performance of a learning algorithm.

1. Training Error: The training error measures how well the algorithm performs on the training set. It is calculated as the percentage of instances in the training set that the algorithm misclassifies or predicts incorrectly. The training error gives an indication of how well the algorithm has learned from the training data.
2. Cross-Validation Error: The cross-validation error measures how well the algorithm generalizes to new, unseen data. It is calculated by evaluating the algorithm's performance on a separate validation set that is not used during training. The cross-validation error helps assess how well the algorithm will perform on new data and gives an indication of its ability to generalize.

The key difference between the two is that the training error measures performance on the data the algorithm has already seen and learned from, while the cross-validation error measures performance on new, unseen data. The goal is to minimize both errors, but it's important to strike a balance between the two. If the training error is significantly lower than the cross-validation error, it suggests that the algorithm may be overfitting the training data and not generalizing well. On the other hand, if the cross-validation error is much higher than the training error, it indicates that the algorithm may be underfitting the data and not capturing important patterns.

By comparing the training error and cross-validation error, you can gain insights into whether the algorithm has a high bias problem (if the training error is much higher than the cross-validation error) or a high variance problem (if the cross-validation error is much higher than the training error).

The training error and cross-validation error can provide insights into whether an algorithm has a high bias or high variance problem. Here's how they can help determine the presence of bias or variance:

1. High Bias: If the training error is significantly higher than the baseline level of performance (such as human level performance) and the cross-validation error is also high but close to the training error, it suggests a high bias problem. This means that the algorithm is not able to capture the underlying patterns in the data, resulting in underfitting. The algorithm is too simplistic and has not learned enough from the training data.
2. High Variance: If the training error is relatively low compared to the baseline level of performance, but the cross-validation error is significantly higher than the training error, it indicates a high variance problem. This means that the algorithm is overfitting the training data, capturing noise and irrelevant patterns. It fails to generalize well to new, unseen data.

By comparing the training error, cross-validation error, and the baseline level of performance, you can assess whether the algorithm is suffering from bias or variance. If the training error is high relative to the baseline level, it suggests bias. If the cross-validation error is high relative to the training error, it suggests variance.

Understanding whether an algorithm has high bias or high variance is crucial for making improvements. High bias can be addressed by using more complex models or increasing the model's capacity. High variance can be reduced by gathering more training data, regularization techniques, or simplifying the model.

Remember, the goal is to strike a balance between bias and variance to achieve optimal performance on new, unseen data.

If the training error is significantly higher than the baseline level of performance, it suggests that the learning algorithm is not able to capture the underlying patterns in the data effectively. Here are the implications of such a scenario:

1. High Bias: When the training error is much higher than the baseline level of performance, it indicates a high bias problem. This means that the algorithm is underfitting the data, failing to capture the complexity and nuances present in the dataset. The algorithm is too simplistic and lacks the capacity to learn from the training data adequately.
2. Insufficient Learning: The algorithm's performance is limited, and it struggles to generalize well to new, unseen data. It may not be able to handle complex patterns or variations in the data, resulting in poor predictive or classification accuracy.
3. Potential Underperformance: If the training error is significantly higher than the baseline level of performance, it suggests that the algorithm is not meeting the desired level of performance. It may not be able to achieve the accuracy or effectiveness expected for the given task.

To address this situation, you may need to consider adjusting the algorithm or model. Some potential steps to improve performance include using more complex models, increasing the model's capacity, or incorporating additional features or data to enhance the algorithm's ability to capture the underlying patterns.

It's important to strike a balance between bias and variance to achieve optimal performance. By iteratively adjusting the algorithm and evaluating its performance, you can work towards reducing the gap between the training error and the baseline level of performance.
