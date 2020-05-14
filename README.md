# Multi-label image classification using CNNs

Movie posters are a key component in the film industry. It is the primary design element that captures the viewer's attention and conveys a movie's theme. Human emotions are aroused by colour, brightness, saturation, hues, contours etc in images. Therefore, we are able to quickly draw general conclusions about a movie's genre (comedy, action, drama, animation etc) based on the colours, facial expressions and scenes portrayed on a movie poster. This leads to the assumption that the colour information, texture features and structural cues contained in images of posters, possess some inherent relationships that could be exploited by ML algorithms for automated prediction of movie genres from posters.  

In this project, CNNs are trained on movie poster images to identify the genres of a movie from its poster. This is a multi-label classification task, since a movie can have multiple genres linked to it, i.e have an independent probability to belong to each label (genre).

The implementation is based on Keras and Tensorflow.

<!-- ![pic1](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Pianist.jpg)
![pic2](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Life-of-Pi.jpg)
![pic3](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Matrix.jpg)
![pic4](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Gladiator.jpg)
![pic5](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/i-robot.jpg) -->

![pic1](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Images/Demo.png)
```
The Pianist (2002)                Family (0.614), War (0.428), Music (0.404)
Life of Pi (2012)                 Adult (0.718), Animation (0.638), Family (0.291)
Gladiator (2000)                  Family (0.736), Adult (0.385), War (0.234)
I-Robot (2004)                    Family (0.495), Crime (0.311), Sci-Fi (0.148)
```

## Requirements
- Keras (with Tensorflow backend)
- pandas
- matplotlib
- numpy
- OpenCV (cv2)

## Dataset

A publicly available IMDB dataset on [Kaggle](https://www.kaggle.com/neha1703/movie-genre-from-its-poster) is used in this project.

It consists of a CSV file ```MovieGenre.csv``` containing the IMDB Id, IMDB Link, Title, IMDB Score, Genre and link to download movie posters.  

![pic1](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Images/original-csv.png#center)

At the time of writing this document (May 2020), information for 40, 108 posters was found.

## Step 1 : Downloading of posters and data preparation

Using the poster download links in ```MovieGenre.csv```, images are downloaded with the ```urllib.request.urlopen()``` function.

Few data cleaning checks are performed :  
- Dropping of dataset rows containing null values for any column (IMDB Id, IMDB Link, Title, IMDB Score, Genre, link)
- Removal of corrupt or bad images
- Removal of duplicate entries (in movie IMDB Id)
- Dropping of dataset rows for which the poster image was not found (broken or missing download link)

Code in ```Get_data.ipynb``` and ```Clean_data.ipynb``` saves images in a folder ```"Posters/"``` and performs the data cleaning steps. To preserve the relationship between poster images and the corresponding movie information, files are saved using the IMDB Id value. For example, ```114709.jpg``` is the poster for movie having IMDB Id 114709.

## Step 2 : Multi-hot encoding of labels
The class labels (i.e the genres) are categorical in nature and have to be converted into numerical form before classification is performed. One-hot encoding is adopted, which converts categorical labels into a vector of binary values. 28 unique genres are found and each genre is represented as a one-hot encoded column. If a movie belongs to a genre, the value is 1("hot"), else 0. As an image can belong to multiple genres, here it is a case of multiple-hot encoding (as multiple genre values can be "hot"). After transformation, the encoded labels look like this:

![multi-hot](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Images/multi-hot-encoding.png)

Code in ```Organise_data.ipynb``` performs this transformation, manually. Alternately, [```MultiLabelBinarizer```](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) from ```sklearn``` can be used for performing one-hot encoding for multi-label data.

The data is partitioned and organised as follows :
- ```Train.csv``` - 25, 842 images
- ```Val.csv``` - 9, 967 images
- ```Test.csv```- 1, 108 images

## Step 3 : Training CNN models

Two CNN architectures were tested in this project. In both cases, the final output layer has 28 neurons (for 28 genre labels) and uses a ```sigmoid```activation function. Unlike softmax activation, the sum of the class-wise predicted probability for a sigmoid network may not necessarily be 1, which enables us to perform multi-label classification.  

- **Method 1 : Custom architecture**

```
num_classes = 28  

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(200,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
```

![pic](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Images/model-summary.png)

Model training parameters :
- Epochs = 30
- Loss = ```binary_crossentropy```
- Optimizer = ```Adam ```
- Input shape = ```(200,150,3)```

Code in ```Train-1.ipynb```

- **Fine-tuning pre-trained VGG16**

VGG16 model is loaded with pre-trained weights (Imagenet) and without the classifier layer (top layer). All the layers, except the last 4, are then frozen. Finally, to this vgg convolutional base model, a fully connected classifier layer is added followed by a sigmoid layer with 28 outputs. For an input image shape of ```(200,150,3)```, the model summary is:

![vgg](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Images/model_summary-vgg.png)

An ```ImageDataGenerator``` is prepared before training, to perform data augmentation.

Model training parameters :
- Epochs = 50
- Loss = ```binary_crossentropy```
- Optimizer = ```RMSprop```

Code in ```Train-2.py```

More details on fine-tuning a pre-trained network in Keras can be found in tutorials [here](https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/) and [here](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html)

## Inference & Accuracy

For predictions, the genres corresponding to the top 3 probability, are chosen.

- Single image prediction in ```Test_single_image.ipynb```
- Evaluating overall accuracy of all test images in ``` Test-accuracy.ipynb```

To compute accuracy, an input image poster is considered as *correctly identified* if atleast 1 out of the 3 predicted genres, are found in the original set of genres for the movie. By this evaluation method, an overall test accuracy of 77.50% is achieved with the fine-tuned vgg16 model and 72.36% with the custom CNN model.

## Further work

This dataset is challenging since it is highly imbalanced. The distribution of images across all genres is :

![genre](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Images/genres.png)

Genres such as ```Drama``` and ```Comedy``` have a high number of instances compared to others, whereas some like ```Talk-Show, Reality-Show``` or ```Game-TV``` have very low instances. Re-sampling techniques could be used for having a more equal distribution of the different genres.

## Acknowledgements

- Tutorial in [Pytorch](https://www.learnopencv.com/multi-label-image-classification-with-pytorch/) and [Keras](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/)
- Blog [post](https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/) and github [repo](https://github.com/benckx/dnn-movie-posters) on poster genre classification
