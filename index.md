## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/manginoja/455/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)


```
### Summary

This project consists of a bird classification model made for a Kaggle competition.  The model itself is a residual neural network built from a modified ResNet50 model, trained using a variety of machine learning techniques including learning rate annealing, weight decay, and data augmentation.  The dataset that the model was trained and tested on consists of over 38,600 images of 555 different species of birds; the goal of training was to achieve the highest top-1 accuracy on correctly classifying a bird species. The final model was able to achieve a training accuracy of BLANK on 34,700 files, and a testing accuracy of BLANK on 3,860 files.

### Problem Setup

This model was made to be submitted to the Kaggle competition “Birds! Are they real?”.  The goal of this competition was to create a model that can classify bird species using the provided images and labels.  The model was then tested on a pre-defined dataset provided through the Kaggle competition, under the folder /test/.  The goal of this model was to provide the highest testing accuracy on this dataset.  

### Dataset

As this was for a Kaggle competition, this model used the dataset provided through the Kaggle site, under the /train/ folder. This folder was pre-organized into sub-folders, each containing a specific species of bird. This was in the format needed to process this data into a dataset using PyTorch’s dataset.ImageFolder function. The dataset was separated into an 80/10/10 train/validation/test split, along with being resized to 224 x 224 images in order to match the size of the images ResNet was trained on. This resizing was done with the goal that ResNet wouldn't have to relearn the sizes of the images from this dataset, allowing it to converge to a solution faster.

### Techniques

This model was primarily a modification of ResNet50 using pretrained weights as a starting point, provided through the torchvision library. The ResNet50 model was modified by stripping the last layer and appending a Linear layer with an output dimension of 555, matching the number of bird classes. I began training without data augmentation or using weight decay.  This resulted in a stark contrast between training accuracy and testing accuracy (0.93 and 0.55 after 30 epochs). In order to fix this apparent overfitting, I chose to add data augmentation by taking random 112 x 112 crops and performing random horizontal flips on the training data, along with normalizing the entire dataset around a mean and standard deviation of 0.5 for all three channels.  I also added a weight decay of 0.0005 during all training epochs. 

To find the optimal learning rate, I chose to use the fastai Learner class and run the lr_find() function before the first epoch of training. This function provides a suggestion for the best learning rate given the model's current state and the weight decay. This yielded the following results:

Learning Rate Suggestion Graph, given by lr_find():
![1-5](https://user-images.githubusercontent.com/36826929/158666067-5d6f6eb9-3179-4bc0-9a04-b370771d705c.png)

*Epochs 1-5*
  -Learning Rate: 0.0025 
  -Beginning Loss: 0.064 
  -End Loss: ~0.028 
  -End Training Accuracy: 0.398 
  -End Testing Accuracy: 0.336

Loss over epochs (blue=training, orange=validation)

![1-5 loss](https://user-images.githubusercontent.com/36826929/158667908-ed28fc19-7130-4e4f-abb2-48326a1ad6ae.png)

Epochs 6-35:

Learning Rate: 0.001
Beginning Loss: 0.028
End Loss: 0.010
Training accuracy: 0.724205
Testing  accuracy: 0.557313

Loss over epochs (blue=training, orange=validation)
![6-35 loss](https://user-images.githubusercontent.com/36826929/158872157-b4ec6d16-de39-4039-9d22-258f046531ae.png)

Epochs 36-52:

Learning Rate: 0.0005
Beginning Loss: 0.010
End Loss: 
Training accuracy: 0.750170
Testing  accuracy: 0.757002

Loss over epochs (blue=training, orange=validation)
![36-54](https://user-images.githubusercontent.com/36826929/158872176-9dfa0aa6-6bdc-477f-8f38-cb45857c57aa.png)


### Additional Info

For this project, the majority of the code was taken from the CNN PyTorch tutorial given by Professor Redmon. This code was mainly just the boilerplate training code along with data augmentation. The code that I implemented was the validation set portion of the training code; using the fastai library to find the optimal learning rate; modifying the data augmentation to match the dataset used; and I had to do research to figure out how to strip the last layer of the resnet50 model. I also wrote the code needed to transform the bird data into usable datasets and dataloaders, along with deriving the number of classes. The rest of the work that went into this project was primarily troubleshooting and then doing training to find the learning rate / decay combination that would yield the best accuracy, along with doing research as to how to prevent overfitting, such as through data normalization and better augmentation.

### References

https://www.kaggle.com/general/74235

https://www.pluralsight.com/guides/introduction-to-resnet (used to strip ResNet of its last layers)

Joe Redmon's CNN PyTorch tutorial (used for training function and data augmentation code).

https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0 (used to find the optimal LR using fastai)

https://docs.fast.ai/tutorial.siamese.html

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).


