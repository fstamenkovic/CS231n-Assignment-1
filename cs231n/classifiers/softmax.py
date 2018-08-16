import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  X_num = X.shape[0]
  num_classes = W.shape[1]
  num_train = X_num

  for i in range(X_num): #Iterate over every single data point
    s = X[i].dot(W) #find the scores matrix
    s -= np.max(s) #subtract the max for numeric stability
    
    curr_loss = -np.log(np.exp(s[y[i]]) / np.sum(np.exp(s))) #find the loss of current
    loss += curr_loss #add to total loss
    p = np.exp(s) / np.sum(np.exp(s))
    p[y[i]] -= 1 
    
    for j in range(num_classes):
      dW[:,j] += X[i,:] * p[j] #update the gradient
    #compute the gradient on scores
    
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  #Regularization 
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W  
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0] 


  s = X.dot(W) #compute scores for every example
  s -= np.max(s, axis = 1, keepdims = True) #remove the max for each example
  p = np.exp(s) / np.sum(np.exp(s),axis=1,keepdims=True) #find 
  
  loss = np.sum(-np.log(p[range(num_train), y]))
  
  p[range(num_train), y] -= 1
  dW = X.transpose().dot(p)
  #compute and store gradients 
  
  #Regularization
  loss /= num_train
  dW /= num_train
  dW += reg * W 

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
                           
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

