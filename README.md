# Detection of dementia on raw voice recordings from the Framingham Heart Study using deep learning

## Technical Overview:
The model code is written using Pytorch and this iteration of the code assumes usage of a GPU. The code can be adjusted to run with a CPU instead, although it will be noticeably slower. The data we used from the Framingham Heart Study is not publicly available and so users will have to supply their own data inputs to the model. Users will also have to create their own functions to both read their own data and input their data to the models.

The purpose of this repository is to supply the code that we used to generate our results from using digital voice data from the Framingham Heart Study from our LSTM and CNN models. However, since we can't supply the actual data we used, the repository can't be directly used to exactly replicate our results - and thus, users must supply their own input data and generate their own results.

We are actively working on finding a public digital voice data set that could be referred to and used by our models.

## Requirements:
Python 3.6.3 or greater. 

Requirements listed below (contained in requirements.txt)  
astroid==2.4.2  
certifi==2020.6.20  
cycler==0.10.0  
fastrlock==0.5  
future==0.18.2  
isort==5.5.3  
joblib==0.16.0  
kiwisolver==1.2.0  
lazy-object-proxy==1.4.3  
matplotlib==3.3.2  
mccabe==0.6.1  
mutagen==1.45.1  
numpy==1.19.2  
pandas==1.1.2  
Pillow==7.2.0  
pylint==2.6.0  
pyparsing==2.4.7  
python-dateutil==2.8.1  
python-speech-features-cuda==0.0.10  
pytz==2020.1  
scikit-learn==0.23.2  
scipy==1.5.2  
six==1.15.0  
sklearn==0.0  
threadpoolctl==2.1.0  
tinytag==1.4.0  
toml==0.10.1  
torch==1.6.0  
tqdm==4.49.0  
typed-ast==1.4.1  
wrapt==1.12.1  

## Installation:
`pip install -r requirements.txt`

## Running the code:
Please run `python train.py -h` to see the various command line arguments that are used to run the model.

## Files and code that must be supplied by the user:
To reiterate: The data we used from the Framingham Heart Study is not publicly available and so users will have to supply their own data inputs to the model. Users will also have to create their own functions to both read their own data and input their data to the models.
### Task CSV text file
A text file that contains the CSV inputs that contain the input data must be supplied by the user. select_task.py shows one way of reading in the CSV path from the task csv text file.
### Select task function [select_task.py]
select_task.py shows how our current code reads our task CSV text file and supplies the CSV input accordingly. The user must supply their own task CSV text file and they can also supply their own select_task() function(s).
## Getting data and labels from the input CSVs [data.py]
The initializer for AudioDataset() contains many attributes that define various functions that are used to read a given input CSV and supply the intended data to the model. The current code defaults all of these various functions to the functions that we used, which are only appropriate to our specific input CSVs (fhs_split_dataframe.py).

Users must change the functions related to querying the input CSVs and getting the label from the input CSVs. The functions related to generating the cross validation folds can likely be reused, but users can adjust them as preferred.

## Viewing model performance
### multi_curves.py
usage: `python multi_curves.py <directory_of_cnn_results> <directory_of_lstm_results>`
This script assumes that the first directory is a directory of CNN results and the second directory is a directory of LSTM results. It will generate a roc_auc curve with both the CNN and LSTM performances and a precision recall curve with both the CNN and LSTM performances. Other performance metrics will be printed out as well.
