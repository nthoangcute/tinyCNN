# tinyCNN
> Very simple convolutional neural network by Java
## INSTALLING
Download the MNIST or EMNIST dataset from [here](https://drive.google.com/drive/folders/10MfF2F5M40NxEFLSpaHWCMo4y8yEMivI?usp=sharing).

Change your path in **trainFile** and **testFile**. 
```
  public String trainFile = "/home/vietbt/java/mnist_digits_train.txt";
  public String testFile = "/home/vietbt/java/mnist_digits_test.txt";
  public double learningRate = 0.55;
  public int batchSize = 50;
  public int outputSize = 10;
```
You also can config the *learningRate*, *batchSize* or *outputSize* to match your dataset.
More information about MNIST or EMNIST datasets is in [here](https://www.nist.gov/itl/iad/image-group/emnist-dataset). 
After that, run this code with your Java IDE or by linux command line:
```
  javac tinyCNN.java
  java tinyCNN
```
## PERFORMANCE
This program runs with all CPUs (updated).

### MNIST Digits Dataset

* Best test accuracy: 99.13% with learning rate = 0.55 after 142,500 steps

<p align="center"><img src="https://lh3.googleusercontent.com/qVY7OSXFDewnnyCHicUBCyMz0jG5oRG6xWNUuPPDu6z-T4F-HeNinKQuEYmclJUllXsq1l8xi_6UBDj6wLZGSvOomzX1-UC0XhyB91Gd1vzdDm58RG8BqSolmXbbM9U_TrcDDUZRfw=w2400" width="600"></p>

### EMNIST Digits Dataset

* Best test accuracy: 98.58% with learning rate = 0.87 after 60,200 steps

<p align="center"><img src="https://lh3.googleusercontent.com/Um9oXvMr7MfPqD8DfILK9GWyydZefh-arTSc3bbm8X0PQssWgpQ7nFFPGzUc05bJ-Uwhc8wPQjrTBfMVP9R5jSfDsfy54-Eu3KirT9WfPLQWS9HVHmJNcWJgiUaVg7cS2RkobtMgtg=w2400" width="600"></p>

### EMNIST Letters Dataset

* Best test accuracy: 88.56% with learning rate = 0.6 after 63,100 steps

<p align="center"><img src="https://lh3.googleusercontent.com/sqFIOgEw4ineHnPgu7uGvU8_u-4OYG8Tp6FvXsO9D41bVOm3na5ZmBkgVLv4wPtEK_M0FMMlEc1lbq2rY85VBN2ZJROtUC02z_HIMNbwe3prG1dROATXfOXd9fowpoQWbKFn1NgBDA=w2400" width="600"></p>

### EMNIST Balanced Dataset

* Best test accuracy: 83.81% with learning rate = 0.6 after 117,300 steps

<p align="center"><img src="https://lh3.googleusercontent.com/eoOr2B8Wn3yBeL2ydc1qKzXsA7oPLNLqbhWvC3x7SZvYXUjrMkMfnhtqhGjGwav0sFMuOee00veG8z84recyDbbYd0J_9-xuVW1pYxwctFJHjLgVKkXBWAvbvw-FU0iOvnx3U6wL0g=w2400" width="600"></p>

### EMNIST By_Merge Dataset

* Best train accuracy: 82.18% with learning rate = 0.85 after 191,300 steps
* Best test accuracy: 80.13% with learning rate = 0.85 after 160,000 steps

<p align="center"><img src="https://lh3.googleusercontent.com/SUz9DW48Uqa6hxilpxtVqD0gSlsLUUVhz49nAVZh5Fmml0qGgUvfLIm8ooG8NkH4beRPrji36OLXBwzryX3-8xRmRCr28j89eBnDlO_vYty6w_0y5O-CHdsglG-spCEb0-d1H8aX3g=w2400" width="600"></p>


### EMNIST By_Class Dataset

* Best train accuracy: 87.00% with learning rate = 0.85 after 206,900 steps
* Best test accuracy: 85.73% with learning rate = 0.85 after 210,000 steps

<p align="center"><img src="https://lh3.googleusercontent.com/AzW5ODSLWLkxl5EY2I4ZFgT4A6TR2LoZXu2f5DswWfgOt0QpBhhXMyDJHNQA2tqaP_-qKsCM1vo8ZygJAUWSuWz4QChuBem4k6Z58QZfuY2dtOW68UeRZOkOeSAxxeEfIeJZxc7cig=w2400" width="600"></p>

## AUTHOR
> **Bui The Viet** - *FPT University* - vietpro213tb@gmail.com
