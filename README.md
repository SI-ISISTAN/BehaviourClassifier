# BehaviourClassifier

Multi-stage behaviour classifier for text messages, expressed as raw text or within a CSV file. The main feature is the 
prediction of a behaviour for each message, using a user-modelling theory called Interaction Process Analysis [1]. With
the prediction of the behaviour detected in the interactions of a given user, it is possible to automatically produce 
collaborative user profiles. This enables and enhances the efficiency of group arrangements.

## Model Architecture

The IPA model establishes a series of *12 categories of behaviour*, which can be grouped into **4 reactions: Positive, 
Negative, Asks and Answers**. This grouping provides the possibility of classifying the behaviour in two stages, using five
classifiers. The first classifier (first stage) predicts a *reaction*, while the remaining ones (second stage) predict the 
final behaviour, given the reaction.

![Classifier Architecture](https://raw.githubusercontent.com/francisco-serrano/BehaviourClassifier/master/classifier_architecture.jpg)

## Data Representation

The text messages are mapped to *sentence-embeddings*. Given a message, it is splitted in words, where each of them
is mapped to a pre-trained word-embedding [2], e.g., a 5-word message, is mapped into 5 word-embedding. Then these
vectors are averaged, producing a final vector (sentence-embedding) which represents the original sentence.

![Classifier Architecture](https://raw.githubusercontent.com/francisco-serrano/BehaviourClassifier/master/sentence_embedding.jpg)

(ilustrative example)

## Deep Learning Approaches

Experiments were made with four types of neural networks, which vary in complexity and results. The best metrics
were achieved with the last approach, based on *Convolucional-Recurrent Neural Networks*.

	- 1st Approach: Regular Deep Neural Networks based on dense layers
	- 2nd Approach: Convolutional Neural Networks
	- 3rd Approach: LSTM-based Recurrent Neural Networks
	- 4th Approach: Hybrid model based on Convolution followed by a Recurrent (CRNN)

## Bibliography
 
[1] Bales, R. F. (1950). Interaction process analysis; a method for the study of small groups.

[2] Cristian Cardellino: Spanish Billion Words Corpus and Embeddings (March 2016),
https://crscardellino.github.io/SBWCE/
