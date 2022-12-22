# log-reg-with-SGD
Implementation of logistic regression using stochastic gradient descent algorithm

A homework I made for Pattern Recognition class to implement logistic regression with stochastic gradient descent from scratch, using NumPy for necessary operations and Matplotlib for plotting the convergence curve. You can see the PY file for the code.

I implemented logistic regression with stochastic gradient descent algorithm using a time-based learning rate schedule (adopted from https://bit.ly/35SCcXO). I tried three different initial learning rates (n = [0.001, 0.01, 0.1]) and three different decay values (d = [0.001, 0.01, 0.1]) on the training data with 100 epochs. Convergence for each combination happened as plotted:

![Convergence curve](https://user-images.githubusercontent.com/76096096/209063266-e889fe89-20d2-4fd2-835b-b67eab478542.png)

I chose n = 0.001 and d = 0.001 as these hyperparameters provided the smoothest convergence, without any fluctuations. Here are the results:

Training accuracy: 97.758

Test accuracy: 95.519
