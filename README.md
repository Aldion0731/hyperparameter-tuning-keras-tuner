### Keras-Tuner Hyperparameter Tuning for Dense Neural Network


Neural networks often contain a large number of hyperparameters that significantly impact their performance. Manually tuning these hyperparameters can be a time-consuming and tedious process. [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) provides an automated solution to efficiently search for the best hyperparameters through various optimization algorithms.

This project utilizes [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) to perform hyperparameter tuning for a dense neural network, and compares the results with results produced by a baseline model. The objective is to optimize the performance of the neural network by finding the best combination of a few selected hyperparameters. The `Hyperband` algorithm is utilized as the optimization algorithm for this project.


### Installation

To run this project, please follow these steps:

- Clone the repository: 
```bash
git clone git@github.com:Aldion0731/hyperparameter-tuning-keras-tuner.git
```
- Install the required dependencies: 
```bash
pipenv sync
```
- Run the code in the notebook located at `src/notebooks/hyperparameter_tuning.ipynb`
### Results

The performances on the baseline and optimized models are shown below

![Results](/results/evaluations.PNG)

We can clearly see that by automated tuning of just a few hyperparamters, we can produce a model that is more accurate and uses significantly less computational resources. We can also conclude that `keras-tuner` offers significant potential in hypertuning.