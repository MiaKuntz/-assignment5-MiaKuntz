# importing pipeline
from transformers import pipeline
# data processing tools
import os
import pandas as pd
# importing plotting tool
import matplotlib.pyplot as plt

# defining function for classifying data
def clf_data(dialog_df): 
    # choosing only "Dialog" column
    dialog = dialog_df["dialog"].values.tolist()
    # using the model emotion english for text classification pipeline
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        # getting the top score for all dialog
                        return_all_scores=False)
    # classifying all dialog and creating pandas series
    emotions = [result["label"] for result in classifier(dialog)]
    emotions = pd.Series(emotions, index=dialog_df.index, name="Emotion")
    return emotions

def plot_emotions(input_emotions, title):
    # plot for dialog emotion distribution
    # creating dictionary
    emotions_dict = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0}
    # counting instances across dialog categories
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
        if file_name.endswith(".csv") and file_name != "movies.csv":
            # creating file paths
            data_file = os.path.join(input_dir, file_name)
            out_file = os.path.join(output_dir, f"{file_name.split('.')[0]}_emotion_dialog_table.csv")
            model_file = os.path.join(models_dir, f"{file_name.split('.')[0]}_emotion_dialog_bars.png")
            # reading in data
            dialog_df = pd.read_csv(data_file)
            # classify emotions
            emotions = clf_data(dialog_df)
            # creating crosstab tables for each emotion distribution
            emotions_table = pd.crosstab(index=emotions, columns="Count")
            # saving tables to csv files
            emotions_table.to_csv(out_file)
            # plotting histogram for all dialog
            plot_emotions(emotions, f"Distribution of emotions for {file_name.split('.')[0]} dialog")
            plt.savefig(model_file)
            plt.clf()

if __name__=="__main__":
    main()
