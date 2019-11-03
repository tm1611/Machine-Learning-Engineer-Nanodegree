# README

## Tools and Software
The library that is used is called `gluonts` which is built on the deep learning framework `MXNet`. Both have to be installed to reproduce the results of the project.

- MXNet - AWS ML instances / Amazon SageMaker come with pre-installed `MXNet`. Hence, you don't need to install it. Otherwise, use pip install to get mxnet
- gluonts - A development version that includes the `DeepState` and `DeepFactor` is necessary. The gluonts version that I used for the porject is `0.3.4.dev68+ga894aee`. Development versions of the most recent github version can be installed using:

`pip install git+https://github.com/awslabs/gluon-ts.git`

## Replication
I used extensive experiments in my project using different number of epochs, batches_per_epoch, and seeds. To replicate the results of my paper you have to change the parameters in the supplied notebook accordingly (change the seed, epoch, num_batches in the provided functions). Note that seeds may vary depending on whether you are using a CPU or GPU. 
