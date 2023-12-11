# Deep Learning Project - Determine Lodging Scores Using Convolutional Neural Neetwork

The files needed to train the model is:

    LoadImagesAndSave.ipynb
        -preproccesing
    LoadAndMatch.ipynb
        -training
    LoadAndPlotBestModel.ipynb
        -plotting data
 
In that order.

THe LoadImagesAndSave file will load the data and use a dataloader on it. Afterwards it will save the train data validation data and test data in a pth-file.
This will then be loaded in to the LoadAndMatch file which will then use the data for the model training. This is doen to be able to reuse the found model for result and graph generating.
The best model during the training is then saved to the models folder, which can then later be used together with the saved data in the LoadAndPlotBestModel file. THe loss and accuracy are also saved to a file in the resuslts folder.
At last the data saved in the two first files can be loaded into the LoadAndPlotBestModel file to create graphs of the model.

