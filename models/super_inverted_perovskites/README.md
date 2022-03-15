# Data

The trainings data is saved in the folders `lambda-training-sets` and `wlog-training-sets`.
Each file pair `train[i].dat` and `test[i].dat` combined is the complete data set.

# Code

In the `maple` folder is the code from [MAPLE repository](https://github.com/GDPlumb/MAPLE).
With the main method from `maple_test.py` one can train and compute the errors for cross validation. The argument of the
method determines, which target should be training (lambda or wlog).