# CNN backend

This folder contains the compact convolutional backend added for the
`cnn_rnn_react` demo.

Design summary:
- input = short sequence of small frames;
- one shared convolution stage encodes each frame;
- global average pooling keeps the state compact;
- a small projection head emits per-frame features for downstream graph leaves.
