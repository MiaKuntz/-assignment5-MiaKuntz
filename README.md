# Assignment 5 – Classifying Dialog Emotions from the Harry Potter Movies
This assignment focuses on feature extraction using two different Emotion Classifications. The classifications will be done on the ``` Harry Potter Movies```, and the objective is to present the results of the emotion classification in a meaningful way using tables and visualisations. The method is much the same as the above assignment in this paper, but this project will focus on looking at the movie franchise Harry Potter, and I here make the assumption that almost everyone is somewhat familiar with the plot, and whether or not the dialog and its emotion follow the plot as this gets progressively darker the further along in the story we get. 

The project will make use of two different classifiers. Since they contain both similar and different emotions I wish to also be able to compare some of the movies’ results between the two classifications. 

## Tasks
The tasks for this assignment are to:
-	Create a pipeline for Emotion Classifications on every movie’s dialog in the data using ```HuggingFace```.
-	Create both meaningful and readable tables and visualisations for the distribution of emotions across all movies.
-	Compare results to the assumption of the dialog having a higher score of “negative emotion” the further along we get in the plot alongside comparing the output of the two different classifiers. 

## Repository content
The GitHub repository contains four folders, namely the ```in``` folder, where the dataset for this assignment can be found, the ``` models ``` folder, which contains visualisations of the emotion distribution across the movies’ dialog as histograms, the ``` out ``` folder, which contains the tables of emotion distributions across the movies’ dialog, and the ``` src ``` folder, which contains the Python scripts for the emotion classifiers. Additionally, the repository has a ```ReadMe.md``` file, as well ```setup.sh``` and ```requirements.txt``` files.

### Repository structure
| Column | Description|
|--------|:-----------|
| ``` in ``` | Folder containing the files for dialog in the Harry Potter movies |
| ```models``` | Folder containing histograms of emotions |
| ```out``` | Folder containing the emotions crosstab tables |
| ```src```  | Folder containing Python scripts for emotion classification |

## Data
The corpus used for this assignment is the ```Harry Potter Movies``` dataset. The dataset consists of several files and folders containing different information, but as this assignment only makes use of the subfolder ```datasets``` as this is where the files all dialog in the movies are located, only this subfolder will be kept in the repository. To download the data, please follow this link:

https://www.kaggle.com/datasets/kornflex/harry-potter-movies-dataset

To access and prepare the data for use in the script, please; Create a user on the website, download the data, and move it to the ```in``` folder in the repository. It should appear as a folder called ```archive```, which is what will be used in the script.

## Methods
The following is a description of parts of my code where additional explanation of my decisions on arguments and functions may be needed than what is otherwise provided in the code. 

To be able to classify and extract emotions based on the dialog in the dataset, I first read the data and load the classifier. When loading the emotion classifier, please know that the model “j-hartmann/emotion-english-distilroberta-base” was chosen due to previous experience with it in the course, along with it being recommended by the course instructor, and the model “bhadresh-savani/distilbert-base-uncased-emotion” was chosen to have a different emotion classifier to compare to. Furthermore is the argument “return_all_scores” in the classifier set to “False” as this ensures that the model only returns the most likely predicted emotion for each line of dialog, which is assumed to be the correct label.

After creating a Pandas series for all dialog, the script then defines a function that creates visualisations for all of the movies and their emotion distribution. I first create an empty dictionary to contain the different emotions, and then use this to count instances across the different lines of dialog. Lastly, the script generates cross tab tables and histograms for each file, that is each movie, and the emotion for all dialog found in it.

## Usage
### Prerequisites and packages
To be able to reproduce and run this code, make sure to have Bash and Python3 installed on whichever device it will be run on. Please be aware that the published script was made and run on a MacBook Pro from 2017 with the MacOS Ventura package, and that all Python code was run successfully on version 3.11.1.

The repository will need to be cloned to your device. Before running the code, please make sure that your Bash terminal is running from the repository; After, please run the following from the command line to install and update the necessary packages:

    bash setup.sh

### Running the scripts
My system requires me to type “python3” at the beginning of my commands, and the following is therefore based on this. To run the scripts from the command line please be aware of your specific system, and whether it is necessary to type “python3”, “python”, or something else in front of the commands. Now run:

	python3 src/hp_emotion_jh.py

And:

	python3 src/ hp_emotion_bs.py

This will activate the scripts. When running, these will go through each of the functions in the order written in my main function. That is:
-	Defining directories for input and output.
-	Looping over all movie files and excluding the unnecessary file.
-	Defining titles for all output depending on the input file.
-	Creating pandas data frame and classifying emotions to store in it.
-	Creating and saving cross tab tables for each file and their emotions.
-	Plotting and saving histograms for each file and their emotions.

## Results
As this project contains the results of two different classifiers, I will first go over the main results of each emotion classification before comparing these to each other.

The first classifier I ran was the J. Hartmann emotion classifier. The results of this classification show that the most found emotion across all the movies is “neutral” followed by “surprise” and “anger”. As it often is with the “neutral” emotion when included in emotion classification I interpret this as the model classifying all dialog not fitting into the other emotion categories as this emotion, although there is a high possibility that a part of the score could be the result of some dialog being neutral in its tone. 

My main reason for also including a second emotion classifier is that this classifier doesn’t include “neutral” as an emotion, which was the top emotion for all movies when using the J. Hartmann emotion classifier. It also doesn’t contain “disgust” but has instead included “love”, which I found to be an interesting emotion to include, although this emotion is found to have one of the lowest scores along with “surprise”. Instead, when using the B. Savani classifier I found the most dominating emotion for all movies to be “anger” with “joy” being in second place. These results are rather interesting, as the first classifier model found “surprise” to be one of the most used emotions, while the second classifier found it to be one of the least used. “Joy” also got a higher score when using the B. Savani classifier, and it can be assumed that what was classified as “neutral” when using the J. Hartmann classifier was found to be “joy” when using the B. Savani classifier.
Several other comparisons could be made from the output generated by the scripts, but the overall conclusion to bring forth is that the results can vary by a lot when using different classifiers, and it is therefore always a good idea to inspect the emotion classifier before use to make sure it covers the emotions one wishes to have its text classified into. 

When looking at the results of both classifiers the overall emotions and their distributions are more or less the same throughout the movies and their plot. These show that the dialog doesn’t go toward a more darker theme as one would  assume when watching the movies, and this example of using emotion classification can therefore not reveal which movie is was classified on or otherwise reveal anything about the different movies.

## References
Bhadresh Savani, “Distilbert base uncased emotion”. https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion/, 2021.

Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
