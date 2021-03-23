from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train): # N
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes): # C
            if j == y[i]: # true label일 때
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:,j] += X[i] # j에 대해 gradient 저장하는 부분 (누적해야함 --> N번 돌면서 쌓임)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # 전체적으로
    dW += reg * 2 * W # element-wise

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # svm의 loss gradient는 indicator func(whether loss != 0)*x_i

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]

    # loss = max(0, s_j - s_yi + 1)
    scores = X.dot(W) # NXC
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train,-1) # np.arange를 쓰면 뭔가 generator 처럼 되는 듯?
    ## True label의 score를 빼준 다음(그냥 500X1짜리 matrix 만들면 알아서 row의 각 요소에서 다 빼줌)에 1을 다 더해줌
    margin = np.maximum(0, scores - correct_class_scores + 1) # np.max로 하면 에러남
    margin[np.arange(num_train), y] = 0 # true class의 margin = 0
    ## np.max vs. np.maximum

    loss = np.sum(margin) / num_train
    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # margin > 0 인 것에 대해서 gradient가 발생함
    # Gradient는 true score 부분은 -X, 아닌 부분은 X
    margin[margin > 0] = 1 # margin matrix를 아예 count값을 담는 matrix로 바꿔버림
    num_over_zero = margin.sum(axis = 1) # column 별로 합을 구함. --> margin > 0인 것의 총 개수 (1X10의 벡터가 나올 것)
    margin[np.arange(num_train), y] = -num_over_zero # 각 row의 true label column에 num_over_zero를 assign
    # 일단 matrix multiplication으로 dW를 계산함. (X.T and margin) --> 여기서 margin은 계수 matrix (1,0, -num_over_zero)
    
    dW = np.dot(X.T, margin) / num_train # 평균 내주기
    
    # regularization term에 대한 gradient도 계산해서 더해줌.
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW

        
        