### Summary

This project consists of a bird classification model made for a Kaggle competition.  The model itself is a residual neural network built from a modified ResNet50 model, trained using a variety of machine learning techniques including learning rate annealing, weight decay, and data augmentation.  The dataset that the model was trained and tested on consists of over 38,600 images of 555 different species of birds; the goal of training was to achieve the highest top-1 accuracy on correctly classifying a bird species. The final model was able to achieve a training accuracy of 0.750170 on 30,880 files, and a testing accuracy of 0.757002 on 3,860 files, along with scoring a 0.7400 in the Kaggle competition.

### Problem Setup

This model was made to be submitted to the Kaggle competition “Birds! Are they real?”.  The goal of this competition was to create a model that can classify bird species using the provided images and labels.  The model was then tested on a pre-defined dataset provided through the Kaggle competition, under the folder /test/.  The goal of this model was to provide the highest testing accuracy on this dataset.  

### Dataset

As this was for a Kaggle competition, this model used the dataset provided through the Kaggle site, under the /train/ folder. This folder was pre-organized into sub-folders, each containing a specific species of bird. This was in the format needed to process this data into a dataset using PyTorch’s dataset.ImageFolder function. The dataset was separated into an 80/10/10 train/validation/test split, along with being resized to 224 x 224 images in order to match the size of the images ResNet was trained on. This resizing was done with the goal that ResNet wouldn't have to relearn the sizes of the images from this dataset, allowing it to converge to a solution faster.

### Techniques

This model was primarily a modification of ResNet50 using pretrained weights as a starting point, provided through the torchvision library. The ResNet50 model was modified by stripping the last layer and appending a Linear layer with an output dimension of 555, matching the number of bird classes. I began training without data augmentation or using weight decay.  This resulted in a stark contrast between training accuracy and testing accuracy (0.93 and 0.55 after 30 epochs). In order to fix this apparent overfitting, I chose to add data augmentation by taking random 112 x 112 crops and performing random horizontal flips on the training data, along with normalizing the entire dataset around a mean and standard deviation of 0.5 for all three channels.  I also added a weight decay of 0.0005 during all training epochs. 

To find the optimal learning rate, I chose to use the fastai Learner class and run the lr_find() function before the first epoch of training. This function provides a suggestion for the best learning rate given the model's current state and the weight decay. This yielded the following results:

Learning Rate Suggestion Graph, given by lr_find():
![1-5](https://user-images.githubusercontent.com/36826929/158666067-5d6f6eb9-3179-4bc0-9a04-b370771d705c.png)

The final training process had the following results:

*Epochs 1-5*<br/> 
Learning Rate: 0.0025 <br/> 
Beginning Loss: 0.064 <br/> 
End Loss: ~0.028 <br/> 
End Training Accuracy: 0.398 <br/> 
End Testing Accuracy: 0.336<br/> 

Loss over epochs (blue=training, orange=validation)

![1-5](https://user-images.githubusercontent.com/36826929/158882416-2e7f0954-1042-485a-998d-ea669e75da93.png)


Epochs 6-35:
<br/> 
Learning Rate: 0.001<br/> 
Beginning Loss: 0.028<br/> 
End Loss: 0.010<br/> 
Training accuracy: 0.724205<br/> 
Testing  accuracy: 0.557313<br/> 

Loss over epochs (blue=training, orange=validation)
![ahhh](https://user-images.githubusercontent.com/36826929/158912442-16f8ebed-ea3e-4043-9327-8bf8bf705a94.png)

Epochs 36-52:<br/> 

Learning Rate: 0.0005<br/> 
Beginning Loss: 0.010<br/> 
End Loss: <br/> 
Training accuracy: 0.750170<br/> 
Testing  accuracy: 0.757002<br/> 

Loss over epochs (blue=training, orange=validation)
![36-54](https://user-images.githubusercontent.com/36826929/158872176-9dfa0aa6-6bdc-477f-8f38-cb45857c57aa.png)


### Discussion
This project encountered a variety of issues. For the first few days working on the project, the training accuracy was hitting 99%, however the testing accuracy remained at 0.0002. This was very unusual, and I tried to augment many different things in my code to make it work. It wasn't until I had an "aha" moment that the test/ folder from Kaggle was only for the competition, not testing my data.  After this, I changed my technique to instead split the data in the train/ folder into training and testing sets, which worked very well. The other main problem I had was not calling net.eval() or net.train() in the first few runs, which ate up a lot of my training time. After I fixed this, everything ran smoothly and I was able to get the model to where it needed to be.

In the future, I think that there are a lot of improvements that could be made. I would first and foremost just give this model more time to train.  Preferably, I would train it for 90 more epochs, halving the learning rate each time.  I would also like to experiment more with changing how the data augmentation occurs, which resnet model I use, what loss function is best, and what decay is optimal. Of course, there are a lot of different permutations of options here, meaning this would take quite some time to test. However, I haven't achieved the optimal performance of this model, and I think trying out different combinations of ways to prevent overfitting could be quite beneficial, especially considering that this model has the ability to get 99% training accuracy. 

I believe the most novel approach I took with this model was using the fastai framework to find the optimal first learning rate. I found that the learning rates it suggested afterwards weren't giving me the performance I was looking for (they would take way too much time to train, even if they might have given better results).  However, I think that using this approach allowed me to quickly find a good "jumping off point" for training, where I could then set a reasonable, educated learning rate schedule afterwards.  


### Additional Info

For this project, the majority of the code was taken from the CNN PyTorch tutorial given by Professor Redmon. This code was mainly just the boilerplate training code along with data augmentation. The code that I implemented was the validation set portion of the training code; using the fastai library to find the optimal learning rate; modifying the data augmentation to match the dataset used; and I had to do research to figure out how to strip the last layer of the resnet50 model. I also wrote the code needed to transform the bird data into usable datasets and dataloaders, along with deriving the number of classes. The rest of the work that went into this project was primarily troubleshooting and then doing training to find the learning rate / decay combination that would yield the best accuracy, along with doing research as to how to prevent overfitting, such as through data normalization and better augmentation.

### References

https://www.kaggle.com/general/74235

https://www.pluralsight.com/guides/introduction-to-resnet (used to strip ResNet of its last layers)

Joe Redmon's CNN PyTorch tutorial (used for training function and data augmentation code).

https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0 (used to find the optimal LR using fastai)

https://docs.fast.ai/tutorial.siamese.html

### Video



https://user-images.githubusercontent.com/36826929/158913674-9e515cef-cd77-4058-9980-f0b8f6f5502d.mp4


