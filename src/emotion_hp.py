# importing pipeline
from transformers import pipeline
# data processing tools
import os
import pandas as pd
# importing plotting tool
import matplotlib.pyplot as plt

# defining function for classifying data
def clf_data(): 
    # creating filepath
    data_file = os.path.join("in/archive/datasets/.csv")
    # reading in data
    dialog_df = pd.read_csv(data_file)
    # random sample
    dialog_df_sample = dialog_df.sample(n=1000, random_state=1)
    # choosing only "Dialog" column
    headlines = dialog_df_sample["Dialogue"]
    # using the model emotion english for text classification pipeline
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        # getting the top score for each headline
                        return_all_scores=False)
    # classifying all dialog and creating pandas series
    emotions = [result["label"] for result in classifier(headlines.tolist())]
    emotions = pd.Series(emotions, index=dialog_df_sample.index, name="Emotion")
    return emotions

def plot_emotions(input_emotions):
# plot for headline emotion distribution
    # creating dictionary
    emotions_dict = {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0}
    # counting instances across headline categories
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

# defining main function
def main():
    # processing
    emotions = clf_data()
    # creating crosstab tables for each emotions distribution
    emotions_table = pd.crosstab(index=emotions, columns="Count")
    # saving tables to csv files
    emotions_table.to_csv("out/emotion_dialog_table.csv")
    # plotting histogram for all headlines
    plot_emotions(emotions)
    plt.title("Distribution of emotions for all dialog")
    plt.savefig("models/emotion_dialog_bars.png")

if __name__=="__main__":
    main()
