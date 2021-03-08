# Detection of dementia on raw voice recordings from the Framingham Heart Study using deep learning

## Technical Overview:
The model code is written using Pytorch and this iteration of the code assumes usage of a GPU. The code can be adjusted to run with a CPU instead, although it will be noticeably slower. The data we used from the Framingham Heart Study is not publicly available and so users will have to supply their own data inputs to the model. Users will also have to create their own functions to both read their own data and input their data to the models.

## Requirements:
Python 3.6.3 or greater

## Installation:
`pip install -r requirements.txt`

## Running the code:
Please run `python train.py -h` to see the various command line arguments that are used to run the model.

## Files that must be supplied by the user:
### Task CSV text file
A text file that contains the CSV inputs that contain the input data must be supplied by the user. select_task.py shows one way of reading in the CSV path from the task csv text file.
### Select task function
select_task.py shows how our current code reads our task CSV text file and supplies the CSV input accordingly. The user must supply their own task CSV text file and they can also supply their own select_task() function(s).
