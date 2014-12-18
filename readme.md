#### Caffe example

Caffe is awesome but it has a lot of dependencies this just hides them behind a simple interface built as a dll / so file.

It's just a play thing on how to use [caffe](http://caffe.berkeleyvision.org) and provides a simple way to use it in other projects.

At present it just provides a Predict function that takes the file path of an image and returns the classification results based on the pre-trained model of your choice.

#### Build instructions

The repo should sit in the same parent folder as your Caffe repo, e.g. ./Caffe and ./Caffe-wrapper. This is because it references the bin and include folder with the caffe directory. 

It assumes you have already built Caffe using [caffe/jaleigh](https://github.com/jaleigh/caffe).

It currently builds with MS Visual Studio 2012 in 86 and 64 bit configurations. 

#### Issues

Issues and pull requests are welcome, it's not something I'll actively maintain though. It is really just an example of using Caffe to predict images.