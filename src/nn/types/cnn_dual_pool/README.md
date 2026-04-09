# CNN dual-pool backend

This folder contains a compact convolutional backend whose structural identity
is explicit in the type name `cnn_dual_pool`.

Design summary:
- input = short sequence of small frames;
- one shared convolution stage encodes each frame;
- every filter emits both a global average-pooled summary and a global max-pooled summary;
- a projection head consumes the concatenated pooled representation;
- backprop keeps the average-pool and max-pool gradient routes explicit.
