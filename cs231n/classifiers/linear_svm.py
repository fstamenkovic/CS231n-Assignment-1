import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################    
    
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train): #iterate over all training examples
    scores = X[i].dot(W) #find the class scores for example
    correct_class_score = scores[y[i]]
  
    for j in range(num_classes):
      if j == y[i]: #skip correct class
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: #find if it is larger than margin
        loss += margin #update loss
        dW[:,y[i]] -= X[i] #Correct class
        dW[:,j] += X[i] #incorrect class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += W * reg

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################


  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W) #We have the predicted scores for the weights W
  delta = 1.0
  #figure out the margins obtained from the scores
  #margin = scores + correct_class_score + 1
  #we need to extract the correct class scores
  correct_class_scores = scores[np.arange(len(scores)), y]
  margins = np.maximum(0, scores - np.matrix(correct_class_scores).transpose() + delta)
  #correct_class_scores was a list, we need to transpose it to subtract from matrix

  #We don't want to include the margins from the correct class, so we need to 
  #set that to zero
  margins[np.arange(len(scores)), y] = 0
  loss = np.sum(margins, axis = 1)
  loss = np.sum(loss)
  loss /= num_train
  
  loss += reg * np.sum(W * W)

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass

  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train
  dW += W * reg
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
