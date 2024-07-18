# Spam Detection System
## Introduction
This project aims to build a spam detection system using Natural Language Processing (NLP) techniques in order to classify emails as spam or not spam based on the content of the email.
The steps involved in this project are:
1. Train a classifier to identify spam emails.
2. Find out the principal topics of the spam emails.
3. Compute the semantic similarity between the spam emails, to verify the etheroegeneity of the spam emails.
4. Extract from non-spam emails the Organisations mentioned in the emails.

## Models
The models used in this project are:
1. LSTM (Long Short Term Memory) Neural Network with Word Embeddings, and a GRU (Gated Recurrent Unit) Neural Network.
2. LDA (Latent Dirichlet Allocation) Topic Modelling.
3. NER (Named Entity Recognition) using Spacy for Organisations extraction.
