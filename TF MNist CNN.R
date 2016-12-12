# Tensorflow python code converted to R by Carl Erick Hagmann 12/12/16

## A Convolutional Network implementation example using TensorFlow library.
## This example is using the MNIST database of handwritten digits
## (http://yann.lecun.com/exdb/mnist/)
## 
## Author: Aymeric Damien
## Project: https://github.com/aymericdamien/TensorFlow-Examples/

library(tensorflow)
# Import MNIST data
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784L # MNIST data input (img shape: 28*28)
n_classes = 10L # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf$placeholder(tf$float32, shape(NULL, n_input))
y = tf$placeholder(tf$float32, shape(NULL, n_classes))
keep_prob = tf$placeholder(tf$float32) #dropout (keep probability)

# Create some wrappers for simplicity
conv2d<-function(x, W, b, strides=1L){
  # Conv2D wrapper, with bias and relu activation
  x = tf$nn$conv2d(x, W, strides=c(1L, strides, strides, 1L), padding='SAME')
  x = tf$nn$bias_add(x, b)
  tf$nn$relu(x)
}

maxpool2d<-function(x, k=2){
  # MaxPool2D wrapper
  tf$nn$max_pool(x, ksize=c(1L, k, k, 1L), strides=c(1L, k, k, 1L), padding='SAME')
}

# Create model
conv_net<-function(x, weights, biases, dropout){
  # Reshape input picture
  x = tf$reshape(x, shape=c(-1L, 28L, 28L, 1L))
  
  # Convolution Layer
  conv1 = conv2d(x, weights$wc1, biases$bc1,1L)
  # Max Pooling (down-sampling)
  conv1 = maxpool2d(conv1, k=2)
  
  # Convolution Layer
  conv2 = conv2d(conv1, weights$wc2, biases$bc2)
  # Max Pooling (down-sampling)
  conv2 = maxpool2d(conv2, k=2)
  
  # Fully connected layer
  # Reshape conv2 output to fit fully connected layer input
  fc1 = tf$reshape(conv2, shape=c(-1L, 7L*7L*64L))
  fc1 = tf$add(tf$matmul(fc1, weights$wd1), biases$bd1)
  fc1 = tf$nn$relu(fc1)
  # Apply Dropout
  fc1 = tf$nn$dropout(fc1, dropout)
  
  # Output, class prediction
  out = tf$add(tf$matmul(fc1, weights$out), biases$out)
  out
}

# Store layers weight & bias
weights=list()
# 5x5 conv, 1 input, 32 outputs
weights$wc1 = tf$Variable(tf$random_normal(c(5L, 5L, 1L, 32L)))
# 5x5 conv, 32 inputs, 64 outputs
weights$wc2 = tf$Variable(tf$random_normal(c(5L, 5L, 32L, 64L)))
# fully connected, 7*7*64 inputs, 1024 outputs
weights$wd1 = tf$Variable(tf$random_normal(c(7L*7L*64L, 1024L)))
# 1024 inputs, 10 outputs (class prediction)
weights$out = tf$Variable(tf$random_normal(c(1024L, n_classes)))

#biases
biases=list()
biases$bc1 = tf$Variable(tf$random_normal(shape(32L)))
biases$bc2 = tf$Variable(tf$random_normal(shape(64L)))
biases$bd1 = tf$Variable(tf$random_normal(shape(1024L)))
biases$out = tf$Variable(tf$random_normal(shape(n_classes)))

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(pred, y))
optimizer = tf$train$AdamOptimizer(learning_rate=learning_rate)$minimize(cost)

# Evaluate model
correct_pred = tf$equal(tf$argmax(pred, 1L), tf$argmax(y, 1L))
accuracy = tf$reduce_mean(tf$cast(correct_pred, tf$float32))

# Define operation but don't run yet
init <- tf$initialize_all_variables()
# Launch model in Session and initialize variables
sess <- tf$InteractiveSession()
sess$run(init)
# Launch the graph
#with(tf$Session() %as% sess, {
  #sess = tf$Session()
  #sess$run(init)
  step = 1
  # Keep training until reach max iterations
  while (step * batch_size < training_iters){
    batch = mnist$train$next_batch(batch_size)
    batch_x<-batch[[1]]
    batch_y<-batch[[2]]
    # Run optimization op (backprop)
    sess$run(optimizer, feed_dict=dict(x=batch_x, y= batch_y,
                                        keep_prob=dropout))
    
    if (step %% display_step == 0){
      # Calculate batch loss and accuracy
      res= sess$run(c(cost, accuracy), feed_dict<-dict(x=batch_x,
                                                       y=batch_y, keep_prob=1L))
      loss=res[[1]]
      acc=res[[2]]
    
    cat(sprintf("Iter %g, Minibatch Loss= %f, Training Accuracy= %f\n"  
            ,step*batch_size, loss, acc))
    }
    step = step + 1
  }
#})
print("Optimization Finished!")
# Calculate accuracy for 256 mnist test images
sprintf("Testing Accuracy: %f", 
      sess$run(accuracy, feed_dict<-dict(x=mnist$test$images,
                                         y=mnist$test$labels, keep_prob=1L))
)
