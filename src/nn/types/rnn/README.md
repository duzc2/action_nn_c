# RNN backend

This folder contains the compact recurrent backend added for the
`cnn_rnn_react` demo.

Design summary:
- input = a short sequence of compact feature vectors;
- one tanh recurrent state integrates short-term temporal context;
- the final hidden state is projected into non-conflicting control axes.
