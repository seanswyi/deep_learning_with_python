# Deep Learning with Python (François Chollet)

_I'll be using this Markdown file to keep notes regarding the book **Deep Learning with Python** written by Keras creator François Chollet._

## Chapter 1: What Is Deep Learning?

### 1.1 Artificial Intelligence, Machine Learning, and Deep Learning

1. You need to be able to extract the signal from the noise (i.e. differentiate between useful information and over-hyped media articles).

2. $DL \in ML \in AI$

#### 1.1.1 Artificial Intelligence

In the 1950's, AI rose as a product of the still new field of computer science when researchers started asking the question of "Is it possible to make computers think like humans?"

The paradigm of "symbolic AI" - basically the `if-else` AI that people joke about - was the dominant paradigm until the 1980's. Thats' when machine learning came into play.

#### 1.1.2 Machine Learning

In the 1830's and 1840's, [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace) and [Charles Babbage](https://en.wikipedia.org/wiki/Charles_Babbage) created the [Analytical Engine](https://en.wikipedia.org/wiki/Analytical_Engine).

The quote by Ada Lovelace, "The Analytical Engine has no pretensions whatever to originate anything. It can be whatever we know how to order it to perform... Its province is to assist us in making available what we are already acquainted with." Basically, the Analytical Engine is just a machine that is programmed by humans. Alan Turing later called this "Lady Lovelace's objection."

This very quote is the essence of machine learning: Can computers do tasks other than what we know? Would they be able to surprise and surpass us? Is it possible for computers to figure out rules on their own?

This is important because in the previous paradigm of symbolic AI, computers would _**receive rules and data as input and output answers**_, but in the machine learning paradigm computers _**receive data and answers as input and output rules**_ (I presonally think this is more supervised learning).

#### 1.1.3 Learning Representation from Data

In order to differentiate between machine learning and deep learning, we need to find out _**what exactly ML algorithms do**_.

The main problem in both ML and DL is to _meaningfully transform data_ or to _learn useful "representation"_ that take us closer to the expected output.

Some problems for ML and DL to solve may become easier if we take a different approach. For example when we want to perform classification on white and black points on a coordinate, performing a _**coordinate change**_ may make the problem easier. This kind of coordinate change is a _new representation_.

#### 1.1.4 The "Deep" in Deep Learning

So, what makes deep learning any different/special?

The "deep" in deep learning basically refers to this idea of multiple layers of representations. Almost always, these "layers" are neural networks.

Deep learning is basically when we have multiple layers of representation. As the data progresses through the network it becomes more and more different from the original - but also more and more useful.

Just think of it as a pipeline of sorts.

#### 1.1.5 Understanding How Deep Learning Works in Three Figures

One goal of training deep learning models is finding the appropriate "weights" that parameterize each layer.

In order to control the output of a neural network, you need to be able to assess how well it's performing. We do this via loss functions (a.k.a. objective functions). Adjustments to the weights are made by the optimizer, which uses the backpropagation algorithm to make adjustments.

#### 1.1.6 What Deep Learning Has Achieved So Far

Just a small section highlighting what kind of tasks have been achieved by deep learning techniques.

#### 1.1.7 Don't Believe the Short-term Hype

"The risk with high expectations for the short term is that, as technology fails to deliver, research investment will dry up, slowing down progress for a long time."

There have been two "AI winters" in the past: one in the 1960's and one in the 1980's. We may be entering another AI Winter soon, and are still in the stages of high expectations and optimism.

It's important to let people know what AI can and cannot do.

#### 1.1.8 The Promise of AI

Brief mention of how AI will impact our future.

### 1.2 Before Deep Learning: A Brief History of Machine Learning

It's important to get familiar with other algorithms so that we don't use the deep learning hammer for everything.

#### 1.2.1 Probabilistic Modeling

Naive Bayes

Logistic Regression

#### 1.2.2 Early Neural Networks

People rediscovered the backpropagation algorithm in the mid-1980's, and Yann LeCun at Bell Labs applied neural nets to handwritten digits. The resulting network was called LeNet and was used by the US Post Office.

#### 1.2.3 Kernel Methods

The most well-known example of kernel methods is the support vector machine (SVM). The SVM utilizes something called the "kernel trick." There are two essential steps in SVMs:

1. map the data to a new high-dimensional representation where the decision boundary can be expressed as a hyperplane

2. try to maximize the distance between the hyperplane and the closest data points

The kernel trick comes into play when we are mapping our data onto higher dimensions.

#### 1.2.4 Decision Trees, Random Forests, and Gradient Boosting Methods

Talks about decision trees, how random forests conduct ensembling for decision trees, and how gradient boosting helps further improve the models by focusing on the weak points of the previous models.

#### 1.2.5 Back to Neural Networks

People performed really well at ImageNet classification using convolutional neural networks. It's become a go-to solution for many problems.

#### 1.2.6 What Makes Deep Learning Different

The reason why deep learning is different from conventional machine learning techniques is due to its ability to automate the process known as "feature engineering."

The reason why feature engineering existed in the first place is due to the limitation of conventional techniques to use raw data to attain satisfactory results. As a result, humans needed to take care of all the data to be fed into the machine.

#### 1.2.7 The Modern Machine Learning Landscape

The two techniques that you should be acquainted with are _**gradient boosting machines (shallow learning**_ and _**deep learning (perceptual problems)**_. An example of each would be XGBoost and Keras.

### 1.3 Why Deep Learning, Why Now?

Many deep learning techniques, such as the LSTM algorithm, have already been with us since the late 1980's and 1990's. Then why has deep learning experienced such a boom in recent years? What has happened?

1. Hardware
2. Datasets and benchmarks
3. Algorithmic advances

#### 1.3.1 Hardware

The development of GPU chips has aided deep learning techniques greatly. In 2007, NVIDA launched CUDA, a programming interface for NVIDIA GPU's.

GPU's are basically supercomputers that specialize in making mathematical computations much more efficiently than CPU's.

#### 1.3.2 Data

"If deep learning is the steam engine of this revolution, then data is its coal."

#### 1.3.3 Algorithms

Algorithms are what allows us to reliably train deep neural networks. In the beginning, DNN's still performed poorly compared to SVM's or Random Forests. The problem was something called the "vanishing gradient problem." Basically, as the number of layers increased, the feedback signal would fade away. This changed with algorithmic improvements.

1. Better "activation functions" for neural layers

2. Better "weight initialization schemes"

3. Better "optimization schemes"

#### 1.3.4 A New Wave of Investment

Basically talks about how much money and people are in AI now compared to the past.

#### 1.3.5 The Democratization of Deep Learning

In the past if you wanted to train a DNN, you needed some expertise in C++ and CUDA, but now you just need to know a bit of Python (or R depending on your preference). 

#### 1.3.6 Will It Last?

The important properties that deep learning models have can be categorized into three categories:

1. _**Simplicity**_: You don't need feature engineering or any other engineering-heavy techniques that conventional machine learning uses.

2. _**Scalability**_: You can easily scale deep learning models by parallelizing GPU's or TPU's.

3. _**Versatility and Reusability**_: You can train DNN's on additional data without having to start from scratch, making them viable for continuous online learning. 