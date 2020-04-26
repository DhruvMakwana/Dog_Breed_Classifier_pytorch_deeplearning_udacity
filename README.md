[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


# Dog_Breed_Classifier_pytorch_deeplearning_udacity

This project is part of deeplearning nanodegree on udacity implemented using pytorch

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!


## Project Instructions

### Dataset used

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  
2. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  

CNN Structures (Building a model on my own)
	(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

	activation: relu

	(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

	activation: relu

	(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

	activation: relu

	(dropout): Dropout(p=0.3)

	(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

	(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

	(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

	(conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

	(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

	(dropout): Dropout(p=0.3)

	(fc1): Linear(in_features=256*3*3, out_features=512, bias=True)

	(dropout): Dropout(p=0.3)

	(fc2): Linear(in_features=512, out_features=133, bias=True)
	
## Results: Accuracy has been achieved to 14% after 25 epochs

## Transfer Learning

Used ResNet50 for transfer learning by changing last layer of pretrained network with adam optimizer at learning rate of 0.001

## Results: Accuracy has been achieved to 81% after 25 epochs

	
	Epoch 1, Batch 1 loss: 4.988276
	Epoch 1, Batch 101 loss: 3.990508
	Epoch 1, Batch 201 loss: 3.050565
	Epoch 1, Batch 301 loss: 2.610464
	Epoch: 1 	Training Loss: 2.497172 	Validation Loss: 0.806791
	Validation loss decreased (inf --> 0.806791).  Saving model ...
	Epoch 2, Batch 1 loss: 1.671902
	Epoch 2, Batch 101 loss: 1.302282
	Epoch 2, Batch 201 loss: 1.275751
	Epoch 2, Batch 301 loss: 1.276304
	Epoch: 2 	Training Loss: 1.271454 	Validation Loss: 0.589309
	Validation loss decreased (0.806791 --> 0.589309).  Saving model ...
	Epoch 3, Batch 1 loss: 0.628817
	Epoch 3, Batch 101 loss: 1.083294
	Epoch 3, Batch 201 loss: 1.088256
	Epoch 3, Batch 301 loss: 1.097040
	Epoch: 3 	Training Loss: 1.083551 	Validation Loss: 0.548926
	Validation loss decreased (0.589309 --> 0.548926).  Saving model ...
	Epoch 4, Batch 1 loss: 1.490823
	Epoch 4, Batch 101 loss: 0.999596
	Epoch 4, Batch 201 loss: 0.997450
	Epoch 4, Batch 301 loss: 0.997139
	Epoch: 4 	Training Loss: 0.997130 	Validation Loss: 0.481983
	Validation loss decreased (0.548926 --> 0.481983).  Saving model ...
	Epoch 5, Batch 1 loss: 0.673107
	Epoch 5, Batch 101 loss: 0.939902
	Epoch 5, Batch 201 loss: 0.933506
	Epoch 5, Batch 301 loss: 0.954241
	Epoch: 5 	Training Loss: 0.956093 	Validation Loss: 0.477505
	Validation loss decreased (0.481983 --> 0.477505).  Saving model ...
	Epoch 6, Batch 1 loss: 0.581581
	Epoch 6, Batch 101 loss: 0.871107
	Epoch 6, Batch 201 loss: 0.862837
	Epoch 6, Batch 301 loss: 0.884670
	Epoch: 6 	Training Loss: 0.884058 	Validation Loss: 0.467822
	Validation loss decreased (0.477505 --> 0.467822).  Saving model ...
	Epoch 7, Batch 1 loss: 1.427830
	Epoch 7, Batch 101 loss: 0.905297
	Epoch 7, Batch 201 loss: 0.910696
	Epoch 7, Batch 301 loss: 0.913614
	Epoch: 7 	Training Loss: 0.906176 	Validation Loss: 0.528639
	Epoch 8, Batch 1 loss: 0.839843
	Epoch 8, Batch 101 loss: 0.880858
	Epoch 8, Batch 201 loss: 0.844327
	Epoch 8, Batch 301 loss: 0.857063
	Epoch: 8 	Training Loss: 0.860199 	Validation Loss: 0.530373
	Epoch 9, Batch 1 loss: 0.713667
	Epoch 9, Batch 101 loss: 0.873187
	Epoch 9, Batch 201 loss: 0.852546
	Epoch 9, Batch 301 loss: 0.875552
	Epoch: 9 	Training Loss: 0.882663 	Validation Loss: 0.610281
	Epoch 10, Batch 1 loss: 1.135839
	Epoch 10, Batch 101 loss: 0.850564
	Epoch 10, Batch 201 loss: 0.843890
	Epoch 10, Batch 301 loss: 0.843047
	Epoch: 10 	Training Loss: 0.859217 	Validation Loss: 0.505544
	Epoch 11, Batch 1 loss: 0.586802
	Epoch 11, Batch 101 loss: 0.786204
	Epoch 11, Batch 201 loss: 0.806350
	Epoch 11, Batch 301 loss: 0.808824
	Epoch: 11 	Training Loss: 0.812037 	Validation Loss: 0.462997
	Validation loss decreased (0.467822 --> 0.462997).  Saving model ...
	Epoch 12, Batch 1 loss: 0.624215
	Epoch 12, Batch 101 loss: 0.786815
	Epoch 12, Batch 201 loss: 0.831431
	Epoch 12, Batch 301 loss: 0.842150
	Epoch: 12 	Training Loss: 0.837809 	Validation Loss: 0.461257
	Validation loss decreased (0.462997 --> 0.461257).  Saving model ...
	Epoch 13, Batch 1 loss: 0.245320
	Epoch 13, Batch 101 loss: 0.774077
	Epoch 13, Batch 201 loss: 0.781466
	Epoch 13, Batch 301 loss: 0.795114
	Epoch: 13 	Training Loss: 0.799339 	Validation Loss: 0.519438
	Epoch 14, Batch 1 loss: 0.350892
	Epoch 14, Batch 101 loss: 0.755855
	Epoch 14, Batch 201 loss: 0.795633
	Epoch 14, Batch 301 loss: 0.802520
	Epoch: 14 	Training Loss: 0.801363 	Validation Loss: 0.479285
	Epoch 15, Batch 1 loss: 1.615883
	Epoch 15, Batch 101 loss: 0.701669
	Epoch 15, Batch 201 loss: 0.729113
	Epoch 15, Batch 301 loss: 0.755251
	Epoch: 15 	Training Loss: 0.774621 	Validation Loss: 0.549827
	Epoch 16, Batch 1 loss: 0.560869
	Epoch 16, Batch 101 loss: 0.814576
	Epoch 16, Batch 201 loss: 0.800165
	Epoch 16, Batch 301 loss: 0.792618
	Epoch: 16 	Training Loss: 0.800380 	Validation Loss: 0.543387
	Epoch 17, Batch 1 loss: 0.142199
	Epoch 17, Batch 101 loss: 0.780947
	Epoch 17, Batch 201 loss: 0.751923
	Epoch 17, Batch 301 loss: 0.768835
	Epoch: 17 	Training Loss: 0.769857 	Validation Loss: 0.481146
	Epoch 18, Batch 1 loss: 0.449748
	Epoch 18, Batch 101 loss: 0.820581
	Epoch 18, Batch 201 loss: 0.782869
	Epoch 18, Batch 301 loss: 0.791180
	Epoch: 18 	Training Loss: 0.780486 	Validation Loss: 0.556208
	Epoch 19, Batch 1 loss: 1.166562
	Epoch 19, Batch 101 loss: 0.820666
	Epoch 19, Batch 201 loss: 0.812832
	Epoch 19, Batch 301 loss: 0.784900
	Epoch: 19 	Training Loss: 0.788714 	Validation Loss: 0.521605
	Epoch 20, Batch 1 loss: 0.761048
	Epoch 20, Batch 101 loss: 0.699008
	Epoch 20, Batch 201 loss: 0.739878
	Epoch 20, Batch 301 loss: 0.759585
	Epoch: 20 	Training Loss: 0.769439 	Validation Loss: 0.587037
	Epoch 21, Batch 1 loss: 0.935080
	Epoch 21, Batch 101 loss: 0.726010
	Epoch 21, Batch 201 loss: 0.771881
	Epoch 21, Batch 301 loss: 0.765983
	Epoch: 21 	Training Loss: 0.770885 	Validation Loss: 0.541668
	Epoch 22, Batch 1 loss: 0.555262
	Epoch 22, Batch 101 loss: 0.728653
	Epoch 22, Batch 201 loss: 0.733303
	Epoch 22, Batch 301 loss: 0.745302
	Epoch: 22 	Training Loss: 0.742251 	Validation Loss: 0.527759
	Epoch 23, Batch 1 loss: 0.294313
	Epoch 23, Batch 101 loss: 0.735635
	Epoch 23, Batch 201 loss: 0.730064
	Epoch 23, Batch 301 loss: 0.758786
	Epoch: 23 	Training Loss: 0.749529 	Validation Loss: 0.520503
	Epoch 24, Batch 1 loss: 0.131095
	Epoch 24, Batch 101 loss: 0.788072
	Epoch 24, Batch 201 loss: 0.754351
	Epoch 24, Batch 301 loss: 0.754937
	Epoch: 24 	Training Loss: 0.759802 	Validation Loss: 0.530764
	Epoch 25, Batch 1 loss: 1.208522
	Epoch 25, Batch 101 loss: 0.643047
	Epoch 25, Batch 201 loss: 0.705212
	Epoch 25, Batch 301 loss: 0.728429
	Epoch: 25 	Training Loss: 0.726061 	Validation Loss: 0.574407

