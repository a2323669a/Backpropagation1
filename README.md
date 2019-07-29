# Backpropagation1
this project is to simulate and realize the Backpropagation(BP) algorithm.
<br><br>
the BP1.ipynb achieved a example of binary classification using simple BP algorithm.
<br>
<br>
the BP2.ipynb achieved a example of binary classification using complete BP algorithm.
In the class Layer, a layer means a neuron layer.
It has two process,forward and backpropagation where we realize the BP algorithm.
We realize two active function relu and sigmoid and its reverse function using in 
the forward and backpropagation process.
<br>
In this example,the loss can reduce to near 6,000 only in about 300 epochs,
compared to the last project jnt3 
nearly 1,0000 in 1000 epochs
<br>
Moverover,we add the stochastic gradient descent way to accelerate the training
process.In fact,it is very well.
<br>
<br>
But we find the overfitting problem is more serious than we considered.
At first, we do not use shuffle and validation.When we using test data(in kaggle),
the accuracy is only about 83 percent,with about 90 percent accuracy in training data.
So,we use shuffle and construct validation data.
```diff
-We use the shuffle wrong.Shuffle is random in the first dim,but we construct the data_x with shape(dim,n).
-In fact we should use the shuffle(data.T)
```
<br>
However, the result is not very well.Even if the accuracy on validation data is over
0.9,the accuracy on test data is most near 0.85, with the epoch only about 20.
<br>
In a word, overfitting is a serious problem.
