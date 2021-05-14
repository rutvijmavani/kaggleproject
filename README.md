# kaggleproject
The dataset is used for this competition is synthetic but based on a real dataset (in this case, the actual Titanic data!) and generated using a CTGAN. 
The statistical properties of this dataset are very similar to the original Titanic dataset, but there's no way to "cheat" by using public labels for predictions. 
How well does your model perform on truly unseen data?  
The data has been split into two groups:  training set (train.csv) test set (test.csv) 
The training set should be used to build your machine learning models. 
For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. 
Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.  
The test set should be used to see how well your model performs on unseen data. 
For the test set, we do not provide the ground truth for each passenger. 
It is your job to predict these outcomes. 
For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Synthanic.
