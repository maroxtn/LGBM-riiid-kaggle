
## What is this ?
This is my LGBM solution for Riiid [kaggle competition](https://www.kaggle.com/c/riiid-test-answer-prediction), this model ensembled with a transformer allowed me to reach the 39th position out of 3395 (2% silver tier).


#### The files are:
- `Main.ipynb`: the main notebook in which I extract all of 70 + features of the model and train it. Running this notebook took me around three days in total, trained on the entire dataset
- `Question correlation.ipynb`: Visualizing the correlation between the 300 most common questions.
- `Test inference`: Notebook in which I tested the inference pipeline, and feature generation from dictionaries on the validation set.
- `create-corr-df.py`: Creating the correlation features data frame which will be joined with the main features data frame.
- `gen-valid-split.py`: Splitting the raw training data into validation folds.
- `generate-corr.py`: Generating questions' correlation data frame
- `heat_pickle`: pickle file containing the data frame of correlation data.



**Used hardware**: Google cloud 256 GB ram, 16 vCPUs. The dataset is really big, and it needs so much memory and expensive hardware.

----
###   Notes on the dataset:
- 100M rows 
- 300k + Users
- 18k questions
For more information, please refer to the challenge page.
