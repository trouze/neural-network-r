## NN in R
## eta=0.1 with 30 epochs should result in about 90% accuracy
## load handwritten digits
source("loadmnist.R")
load_mnist()

## miscellaneous functions
sigmoid <- function(z) {1/(1+exp(-z))}
sigmoid_prime <- function(z) {sigmoid(z)*(1-sigmoid(z))}
digit.to.vector <- function(digit){
  sapply(digit, function(d) c(rep(0,d),1,rep(0,9-d)))
}

## train the neural network using mini batch SGD
## input: MNIST training data, num of epochs, batch size
## learning rate for each epoch
## takes initialized weights and biases
SGD <- function(training_data, epochs, mini.batch.size, eta, sizes, num_layers, biases, weight, verbose=TRUE, test_data){
  ## turn data into matrices
  training_data <- cbind(train$x,train$y) # appends the result to the 785th column, with 60000 rows - one row per observation
  for (j in 1:epochs){
    training_data <- training_data[sample(nrow(training_data)),] # shuffle the data to prep for mini-batches
    mini.batches <- list()
    seq1 <- seq(from=1, to=60000, by=10)
    for(u in 1:(nrow(training_data)/mini.batch.size)){
      # pull out 10 rows from training_data for each iteration of this loop to create 6000 mini-batches
      mini.batches[[u]] <- training_data[seq1[u]:(seq1[u]+9),]
    }
    ## feed forward all mini-batches
    for(g in 1:length(mini.batches)){
      # update weights and biases
      wb <- update.mini.batch(mini.batches[[g]], 0.1, sizes, num_layers, biases, weight, mini.batch.size) 
      # update.mini.batch outputs a list of biases at [[1]] and weights at [[2]]
      biases <- wb[[1]]
      weight <- wb[[2]]
    }
    # Logging
    if(verbose){if(j %% 1 == 0){
      cat("Epoch: ", j, " complete")
      # Print acc and hide confusion matrix
      evaluate(test_data, biases, weight)
    }}
  }
  # return trained weights and biases
  list(biases, weight)
}

## update the network's weights and biases via Gradient descent
## using backpropagation to a single mini batch
## for each epoch we shuffle the training data
## and partition it into mini-batches
update.mini.batch <- function(mini_batch, eta, sizes, num_layers, biases, weight, mini.batch.size){
  ## update globally
  ## instatiate biases and weights to be updated
  nabla.b <- list(rep(0,sizes[2]),rep(0,sizes[3]))
  nabla.w <- list(matrix(rep(0,(sizes[2]*sizes[1])), nrow=sizes[2], ncol=sizes[1]),
                  matrix(rep(0,(sizes[3]*sizes[2])), nrow=sizes[3], ncol=sizes[2]))
  ## train through mini-batch
  for(p in 1:mini.batch.size){
    x <- mini_batch[p,-785]
    y <- mini_batch[p,785]

    ## backprop for each observation in mini-batch
    delta_nablas <- backprop(x, y, sizes, num_layers, biases, weight)
    delta_nabla_b <- delta_nablas[[1]]
    delta_nabla_w <- delta_nablas[[2]]
    ## we must append deltas to nabla - the python code does this much simpler
    ## we want to calculate these per epoch: indexed at j
    nabla.b <- lapply(seq_along(biases),function(j)
      unlist(nabla.b[[j]])+unlist(delta_nabla_b[[j]]))
    nabla.w <- lapply(seq_along(weight),function(j)
      unlist(nabla.w[[j]])+unlist(delta_nabla_w[[j]]))
  }
  ## mini batch is now finished, so we must update weights and biases
  weight <- lapply(seq_along(weight), function(j)
    unlist(weight[[j]])-(eta/mini.batch.size)*unlist(nabla.w[[j]]))
  biases <- lapply(seq_along(biases), function(j)
    unlist(biases[[j]])-(eta/mini.batch.size)*unlist(nabla.b[[j]]))
  # Return biases and weights as we don't have "self" to do this
  return(list(biases, weight))
}

## return tuple representing the gradient for the cost fn C_x
backprop <- function(x, y, sizes, num_layers, biases, weight){
  ## initialize updates
  nabla_b_backprop <- list(rep(0,sizes[2]),rep(0,sizes[3]))
  nabla_w_backprop <- list(matrix(rep(0,(sizes[2]*sizes[1])), nrow=sizes[2], ncol=sizes[1]),
                  matrix(rep(0,(sizes[3]*sizes[2])), nrow=sizes[3], ncol=sizes[2]))
  ## Feed Forward
  activation <- matrix(x, nrow=length(x), ncol=1) # all 784 inputs in single column matrix
  activations <- list(matrix(x, nrow=length(x), ncol=1)) # list to store all activations, layer by layer
  zs <- list() # list to store all z vectors, layer by layer
  

  for(f in 1:length(weight)){
    b <- biases[[f]]
    w <- weight[[f]]
    w_a <- w%*%activation
    b_broadcast <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[2])
    z <- w_a + b
    zs[[f]] <- z
    activation <- sigmoid(z)
    activations[[f+1]] <- activation
  }
  ## backpropagate where we update the gradient using delta errors
  delta <- cost.derivative(activations[[length(activations)]], y) * sigmoid_prime(zs[[length(zs)]])
  nabla_b_backprop[[length(nabla_b_backprop)]] <- delta
  nabla_w_backprop[[length(nabla_w_backprop)]] <- delta %*% t(activations[[length(activations)-1]])
  ## second to second to last layer

  for (q in 2:(num_layers-1)) {
    sp <- sigmoid_prime(zs[[length(zs)-(q-1)]])
    delta <- (t(weight[[length(weight)-(q-2)]]) %*% delta) * sp
    nabla_b_backprop[[length(nabla_b_backprop)-(q-1)]] <- delta
    testyy <- t(activations[[length(activations)-q]])
    nabla_w_backprop[[length(nabla_w_backprop)-(q-1)]] <- delta %*% testyy
  }
  return(list(nabla_b_backprop,nabla_w_backprop))
}

feedforward <- function(a, biases, weight)
{
  for (t in 1:length(biases)){
    a <- matrix(a, nrow=length(a), ncol=1)
    b <- biases[[t]]
    w <- weight[[t]]
    # (py) a = sigmoid(np.dot(w, a) + b)
    # Equivalent of python np.dot(w,a)
    w_a <- w%*%a
    # Need to manually broadcast b to conform to np.dot(w,a)
    b_broadcast <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
    a <- sigmoid(w_a + b_broadcast)
  }
  a
}

classify <- function(test.x, biases, weight)
{
  ## equivalent to running argmax in line 126 of network.py
  lapply(c(1:10000), function(i) {
    which.max(feedforward(test.x[i,], biases, weight))}
  )
}
## returns # of test inputs for which neural network
## outputs the correct result
evaluate <- function(test, biases, weight)
{
  ## manipulate test data like the training data
  test.x <- test[[2]]
  test.y <- test[[3]]
  pred <- classify(test.x, biases, weight)
  truths <- test.y
  # Accuracy
  correct <- sum(mapply(function(x,y) x-1==y, pred, truths))
  total <- 10000
  print(correct/total) # this prints out after each epoch - called in SGD function
}

## return a vector of partial derivatives C_x
cost.derivative <- function(output.activations, y){
  output.activations - digit.to.vector(y)
}

## initialize info to be passed to run NN algo
sizes <- c(784, 30, 10) # inputs
num_layers <- length(sizes)
eta <- 0.1
epochs <- 30
mini.batch.size <- 10
## set biases
biases <- list(rnorm(sizes[2]),rnorm(sizes[3]))
## set weights from node k to node j -> w[j,k]
## we'll use "weight" as there is a function weights()
weight <- list(matrix(rnorm(sizes[2]*sizes[1]), nrow=sizes[2], ncol=sizes[1]),
               matrix(rnorm(sizes[3]*sizes[2]), nrow=sizes[3], ncol=sizes[2]))

## run this to perform the NN algorithm
SGD(train, epochs, mini.batch.size, eta, sizes, num_layers, biases, weight, verbose=TRUE, test)