# Author: Mia Kuntz
# Date hand-in: 31/5 - 2023

# Description: This script classifies the dialog in all CSV files in the input directory using the pretrained model from huggingface model hub (https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) and plots a histogram for each CSV file.
# The histogram shows the distribution of emotions in the dialog. 
# The script saves the crosstab tables and the histograms to the output directory.

# importing pipeline
from transformers import pipeline
# importing system operating tools
import os
# importing pandas
import pandas as pd
# importing plotting tool
import matplotlib.pyplot as plt

# defining function for classifying dialog
def clf_data(dialog_df):
    # choosing only "Dialog" column and converting to list
    dialog = dialog_df["dialog"].astype(str).values.tolist()
    # initializing pipeline for text classification with pretrained model from huggingface model hub (https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
    classifier = pipeline("text-classification",
                          model="bhadresh-savani/distilbert-base-uncased-emotion",
                          return_all_scores=False) # setting return_all_scores to False to only return the label with the highest probability
    # classifying dialog and saving results to list of emotions 
    emotions = [result["label"] for result in classifier(dialog)]
    # creating pandas series from list of emotions
    emotions = pd.Series(emotions, index=dialog_df.index, name="Emotion")
    return emotions

# defining function for plotting histogram
def plot_emotions(input_emotions, title):
    # creating dictionary for counting instances of each emotion in input list 
    emotions_dict = {"love": 0, "joy": 0, "sadness": 0, "anger": 0, "surprise": 0, "fear": 0}
    # counting instances across dialog categories and adding to dictionary
    for emotion in input_emotions:
        emotions_dict[emotion] += 1
    # setting values variable
    values = list(emotions_dict.values())
    # setting names variable
    names = list(emotions_dict.keys())
    # setting arguments for bar plot
    plt.bar(names, values)
    # setting x axis label
    plt.xlabel("Emotion")
    # setting y axis label
    plt.ylabel("Count")
    # setting title
    plt.title(title)

# defining main function
def main():
    # defining input and output directories
    input_dir = "in/archive/datasets"
    output_dir = "out"
    models_dir = "models"
    # looping over all CSV files in the input directory 
    for file_name in os.listdir(input_dir):
        # checking if file is a CSV file and not the movies.csv file
        if file_name.endswith(".csv") and file_name != "movies.csv":
            # creating file paths for input and output files 
            data_file = os.path.join(input_dir, file_name)
            out_file = os.path.join(output_dir, f"{file_name.split('.')[0]}_emotion_dialog_table_bs.csv")
            model_file = os.path.join(models_dir, f"{file_name.split('.')[0]}_emotion_dialog_hist_bs.png")
            # reading CSV file into pandas dataframe
            dialog_df = pd.read_csv(data_file)
            # classify emotions in dialog
            emotions = clf_data(dialog_df)
            # creating crosstab tables for each emotion distribution 
            emotions_table = pd.crosstab(index=emotions, columns="Count")
            # saving crosstab tables to CSV file
            emotions_table.to_csv(out_file)
            # plotting histograms for all dialog
            plot_emotions(emotions, f"Distribution of emotions for {file_name.split('.')[0]} dialog")
            plt.savefig(model_file)
            plt.clf()

if __name__=="__main__":
    main()

# Command line argument:
# python3 hp_emotion_bs.py