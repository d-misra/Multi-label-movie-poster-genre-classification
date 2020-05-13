# Multi-label image classification using CNNs

Movie posters are a key component in the film industry. It is the primary design element that captures the viewer's attention and conveys a movie's theme. Human emotions are aroused by colour, brightness, saturation, hues, contours etc in images. Therefore, we are able to quickly draw general conclusions about a movie's genre (comedy, action, drama, animation etc) based on the colours, facial expressions and scenes portrayed on a movie poster. This leads to the assumption that the colour information, texture features and structural cues contained in images of posters, possess some inherent relationships that could be exploited by ML algorithms for automated prediction of movie genres from posters.  

In this project, CNNs are trained on movie poster images to identify the genres of a movie from its poster. This is a multi-label classification task, since a movie can have multiple genres linked to it, i.e have an independent probability to belong to each label (genre).

The implementation is based on Keras and Tensorflow.

<!-- ![pic1](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Pianist.jpg)
![pic2](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Life-of-Pi.jpg)
![pic3](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Matrix.jpg)
![pic4](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Gladiator.jpg)
![pic5](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/i-robot.jpg) -->

![pic1](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/Demo.png)
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

![pic1](https://github.com/d-misra/Multi-label-movie-poster-genre-classification/blob/master/Poster-images-test/original-csv.png#center)

At the time of writing this document (May 2020), information for 40, 108 posters was found.

### Step 1 : Downloading of posters and data preparation

Using the poster download links in ```MovieGenre.csv```, poster images are downloaded with the ```urllib.request.urlopen()```function.

Few data cleaning checks are performed :  
- Dropping of dataset rows containing null values for any of the columns (IMDB Id, IMDB Link, Title, IMDB Score, Genre or link)
- Removal of corrupt or bad images
- Removal of duplicate entries (in movie IMDB Id)
- Dropping of dataset rows for which the poster image was not found (broken or missing download link)

Code in ```Get_data.ipynb``` and ```Clean_data.ipynb``` saves images in a folder ```"Posters/"``` and performs the data cleaning steps. To preserve the relationship between poster images and the corresponding movie information, files are saved using the IMDB Id value. For example, ```114709.jpg``` is the poster for movie having IMDB Id 114709.

### Step 2 : One-hot encoding of multi-labels
