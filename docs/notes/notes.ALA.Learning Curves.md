---
id: lffa8wbs7bgufdscpxa5n9p
title: Learning Curves
desc: "Coursera Advanced Learning Algorithms bias and variance learning curves"
updated: 1716445760985
created: 1716445330791
---

# Learning Curves

- Learning curves show how the performance of a learning algorithm changes as the amount of training data increases.
- For a model that fits a second-order polynomial function the cross-validation error $J_{CV}$ decreases as the training set size $m_{train}$ increases, indicating that the model is learning better. However, the training error $J_{train}$ increases as the training set size increases. This is because it becomes harder to fit all the training examples perfectly.
- Learning curves can help us identify whether a learning algorithm has high Bias or high variance.
- High bias means the algorithm is underfitting the data, and increasing the training set size won't help much.
- High variance means the algorithm is overfitting the data, and increasing the training set size can help improve performance.
- Learning curves can be used to visualize the performance of a learning algorithm and determine if it has high bias or high variance.

The relationship between the training set size and the cross-validation error can provide insights into the performance of a learning algorithm. Here's how they are related:

1. Small Training Set Size: When the training set size is small, the algorithm has limited data to learn from. In this case, the algorithm may be able to fit the training examples well, resulting in a low training error. However, since the algorithm has not seen much data, it may not generalize well to new, unseen examples. As a result, the cross-validation error may be higher than the training error.

2. Increasing Training Set Size: As the training set size increases, the algorithm has more data to learn from. Initially, the algorithm may be able to fit the training examples well, leading to a decrease in the training error. However, as the training set size continues to increase, it becomes harder for the algorithm to fit every training example perfectly. This can cause the training error to increase. On the other hand, the algorithm may start to generalize better to new examples, resulting in a decrease in the cross-validation error.

3. Large Training Set Size: When the training set size becomes very large, the algorithm has a substantial amount of data to learn from. At this point, the algorithm's performance may plateau, meaning that further increasing the training set size does not significantly improve the model's performance. Both the training error and the cross-validation error may flatten out, indicating that the algorithm has reached its optimal performance given the available data.

In summary, as the training set size increases, the algorithm's performance may improve initially, but there may be a point where further increasing the training set size does not lead to significant improvements. The cross-validation error can help us understand how well the algorithm generalizes to new examples as the training set size changes.

Learning curves provide valuable insights into the performance of a learning algorithm by visualizing how the algorithm's error changes as the training set size increases. Here's how learning curves help us understand algorithm performance:

1. Bias and Variance Analysis:

   - Learning curves can help us determine if a learning algorithm suffers from high bias or high variance.
   - High bias indicates that the algorithm is underfitting the data, meaning it is too simple to capture the underlying patterns. In this case, both the training error and cross-validation error will be high.
   - High variance indicates that the algorithm is overfitting the data, meaning it is too complex and memorizes the training examples instead of generalizing well. In this case, the training error will be low, but the cross-validation error will be significantly higher.

2. Model Complexity Assessment:

   - Learning curves can help us assess the appropriate level of model complexity.
   - If the learning curves for both the training error and cross-validation error converge to a low value, it suggests that the algorithm is performing well and the current model complexity is appropriate.
   - If the learning curves do not converge, it indicates that the algorithm may benefit from increasing or decreasing the model complexity.

3. Estimating Performance:

   - Learning curves can provide an estimate of the algorithm's performance on unseen data.
   - By examining the cross-validation error on the learning curve, we can get an idea of how well the algorithm will perform on new examples.
   - If the cross-validation error is significantly higher than the training error, it suggests that the algorithm may not generalize well to new data.

4. Decision-Making on Data Collection:

   - Learning curves can help us make informed decisions about collecting more training data.
   - If the learning curves indicate high bias, adding more training data may not significantly improve the algorithm's performance. Other approaches, such as increasing model complexity, may be more effective.
   - If the learning curves indicate high variance, adding more training data can help improve the algorithm's performance by reducing overfitting.

In summary, learning curves provide a visual representation of the algorithm's performance, helping us understand if it suffers from bias or variance, assess model complexity, estimate performance on unseen data, and make decisions about data collection. They are a valuable tool for evaluating and improving learning algorithms.
