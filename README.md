
# Gaussian Naive Bayes in Android : Iris Classification Problem 
This project aims to create a Gaussian Naive Bayes classifier in Android and use it on the famous [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). Gaussian Naive Bayes is a different version of commonly used Naive Bayes classifier, as it deals with numerical features.

We assume that each numerical feature in our dataset, follows a Gaussian distribution, with the mean and standard deviation calculated from that feature itself. 

## The Math
Suppose we are given a dataset $D$  with $N_f$ features, where each sample belongs to one of the $N_c$ classes. Our job is to predict the class of a given sample $X$, where $X=\{ x_i \} \ , 1 \leq i \leq N_f$ . 

First, we compute the prior probabilities $p(c_j) \ , 1 \leq j \leq N_c$. The prior probability of class $c_j$ equals the probability that a given sample from our dataset belongs to class $c_j$.

$p(c_j) = \frac{n( c_j )}{n( D )} \ \ \ 1 \leq j \leq N_c$

where $n( c_j )$ is the no. of samples belonging to class $c_j$ and $n( D )$ is the no. of samples in our dataset $D$.

As I mentioned earlier, we assume that each feature follows a Gaussian Distribution, where the likelihood of a feature $x_i$ given a class $c_j$ is,

$L(x=x_i | c_j ) = \frac{1}{\sigma \sqrt{2 \pi}} \exp( - \frac{1}{2} (\frac{x - \mu}{ \sigma})^2)$


$p( c_j|sample ) = p(c_j) \ \prod_{i=1}^N \ L( x=x_i| c_j )$