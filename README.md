# deep-learning-cnn

An implementation of a Convolutional Neural Network on a big image dataset. I used [pytorch](https://pytorch.org) but you can use also different deep layer framework.

The code implements a basic NN and CNN, the data loading, the training phase and the evaluation (testing) phase. The training and testing are on CIFAR 100 dataset (already included in Pytorch).

**Hint:** I suggest you to do not use your NVIDIA card (or other) in your PC. Usually it does not have a well done Deep Learning framework setup. So, you can use [google colab](https://colab.research.google.com/): a very user friendly python notebook from Google in which you can install python packages, download datasets, plot images, and you have a free GPU to do trainings!

# Main concepts

Every NN has three types of layers: input, hidden, and output. Creating the NN architecture therefore means coming up with values for the number of layers of each type and the number of nodes in each of these layers.

### Input layer

Simple--every NN has exactly one of them--no exceptions that I'm aware of.

With respect to the number of neurons comprising this layer, this parameter is completely and uniquely determined once you know the shape of your training data. Specifically, the number of neurons comprising that layer is equal to the number of features (columns) in your data. Some NN configurations add one additional node for a bias term.
  
### Output layer

Like the Input layer, every NN has exactly one output layer. Determining its size (number of neurons) is simple; it is completely determined by the chosen model configuration.

Is your NN going running in Machine Mode or Regression Mode (the ML convention of using a term that is also used in statistics but assigning a different meaning to it is very confusing). Machine mode: returns a class label (e.g., "Premium Account"/"Basic Account"). Regression Mode returns a value (e.g., price).

If the NN is a regressor, then the output layer has a single node.

If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.

### Hidden layer

So those few rules set the number of layers and size (neurons/layer) for both the input and output layers. That leaves the hidden layers.

How many hidden layers? Well if your data is linearly separable (which you often know by the time you begin coding a NN) then you don't need any hidden layers at all. Of course, you don't need an NN to resolve your data either, but it will still do the job.

Beyond that, as you probably know, there's a mountain of commentary on the question of hidden layer configuration in NNs (see the insanely thorough and insightful NN FAQ for an excellent summary of that commentary). One issue within this subject on which there is a consensus is the performance difference from adding additional hidden layers: the situations in which performance improves with a second (or third, etc.) hidden layer are very few. One hidden layer is sufficient for the large majority of problems.

So what about size of the hidden layer(s)--how many neurons? There are some empirically-derived rules-of-thumb, of these, the most commonly relied on is 'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'. Jeff Heaton, author of Introduction to Neural Networks in Java offers a few more.

In sum, for most problems, one could probably get decent performance (even without a second optimization step) by setting the hidden layer configuration using just two rules: (i) number of hidden layers equals one; and (ii) the number of neurons in that layer is the mean of the neurons in the input and output layers. 

# Optimization of the Network Configuration

A CNN is characterized by several parameters, such as: epoch, batch size and learnin rate. 

An **epoch** is a hyperparameter which is defined before training a model. One epoch is when an entire dataset is passed both forward and backward through the neural network only once.

One epoch is too big to feed to the computer at once. So, we divide it in several smaller **batches**. We use more than one epoch because passing the entire dataset through a neural network is not enough and we need to pass the full dataset multiple times to the same neural network. But since we are using a limited dataset and to optimise the learning and the graph we are using Gradient Descent which is an iterative process. So, updating the weights with single pass or one epoch is not enough.

A batch is the total number of training examples present in a single batch and an iteration is the number of batches needed to complete one epoch.

For example: If we divide a dataset of 2000 training examples into 500 batches, then 4 iterations will complete 1 epoch.

The **learning rate** is one of the most important hyper-parameters to tune for training deep neural networks. If the learning rate is low, then training is more reliable, but optimization will take a lot of time because steps towards the minimum of the loss function are tiny.

If the learning rate is high, then training may not converge or even diverge. Weight changes can be so big that the optimizer overshoots the minimum and makes the loss worse.

The training should start from a relatively large learning rate because, in the beginning, random weights are far from optimal, and then the learning rate can decrease during training to allow more fine-grained weight updates.

# Accuracy & Loss

The lower the loss, the better a model (unless the model has over-fitted to the training data). The loss is calculated on training and validation and its interperation is how well the model is doing for these two sets. Unlike accuracy, loss is not a percentage. It is a summation of the errors made for each example in training or validation sets.

In the case of neural networks, the loss is usually negative log-likelihood and residual sum of squares for classification and regression respectively. Then naturally, the main objective in a learning model is to reduce (minimize) the loss function's value with respect to the model's parameters by changing the weight vector values through different optimization methods, such as backpropagation in neural networks.

Loss value implies how well or poorly a certain model behaves after each iteration of optimization. Ideally, one would expect the reduction of loss after each, or several, iteration(s).

The accuracy of a model is usually determined after the model parameters are learned and fixed and no learning is taking place. Then the test samples are fed to the model and the number of mistakes (zero-one loss) the model makes are recorded, after comparison to the true targets. Then the percentage of misclassification is calculated.

There are also some subtleties while reducing the loss value. For instance, you may run into the problem of over-fitting in which the model "memorizes" the training examples and becomes kind of ineffective for the test set. Over-fitting also occurs in cases where you do not employ a regularization, you have a very complex model (the number of free parameters W is large) or the number of data points N is very low.

# Classes

### NN Class

The NN class provides and two hidden layers and one FC layer network. I trained this class on the CIFAR 100 train set.

### CNN Class

A Convolutional Neural Network (CNN) is a class of deep neural networks, most commonly applied to analyzing visual imagery.

### ResNet18 Class

# Useful link

- [NN tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [Data Augmentation](https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=yLEwF_2RzGs0)
