# Description

In this project we tried to translate from German to english. Both the file models_encoder_decoder.py and models_transformers.py does the same thing but using different approach. 

The first one uses the encoder and decoder library of transformer for the neural network. The layers, encoder decoder, positional encoder all has to be defined manually for this task. 

When we used the nn.transformer() in the neural network we didn't have to bother about these encoders, decoders. We only need to put the number of layers, heads etc. It's less customizable.
