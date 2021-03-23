from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        scores = X[i].dot(W) # c차원의 벡터
        correct_class_score = scores[y[i]]
        sum_scores = np.sum(np.exp(scores))
        
        ## cross entropy loss 계산
        ## gradient는 y[i] = j 일 때와 아닐 때를 구분해서 계산
        
        for j in range(num_classes):
            if j == y[i]:
                continue
            softmax = np.exp(scores[j])/sum_scores
            dW[:,j] += X[i]*softmax
        dW[:, y[i]] -= X[i]*(1-np.exp(correct_class_score)/sum_scores)
        loss -= np.log(np.exp(correct_class_score) / sum_scores)
    
    loss /= num_train # avaraging loss
    dW /= num_train
    
    # reg term 더해주기
    loss += reg*np.sum(W*W)
    dW += reg*2*W    
            

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) # DXC

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    
    scores = np.dot(X, W) # score matrix --> NXC 
    softmax = np.exp(scores)/np.sum(np.exp(scores), axis= 1).reshape(num_train,-1) # 분모는 row개수만큼의 summation vector가 나와야 함. --> reshape을 해줘서 matrix 형태로 만들어줘야 broadcast가 가능함. numpy에서는 matrix와 vector 간의 broadcast가 안됨.
    loss = -np.sum(np.log(softmax[np.arange(num_train), y]))/num_train # y_i항만 loss에 반영해야 하므로 np.arange로 인덱싱
    
    softmax[np.arange(num_train), y] -= 1 # 1씩 빼주기 (y_i일 때는 grad가 x_i*(s_yi-1)이니까 )
    dW = np.dot(X.T, softmax)
    dW /= num_train
    
    # reg term
    loss += reg*np.sum(W*W)
    dW += reg*2*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
