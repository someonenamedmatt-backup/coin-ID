# Coin-ID
#### <i> Using Neural Networks to quickly and easily grade rare coins
___

This repository is divided into 3 sections - app, data, and src.  

## 1. app
The app section of the code contains a prototype application for coin-ID. The templates and static section are for cosmetic tweaks while app.py contains the flask app. The predict portion of the code is currently incomplete on the main brach (the trained nets savefiles were to large to upload onto github - please contact me at my [website](mattschiffman.com) if you'd like the trained network.

___

## 2. data
Here you can a few data files used for the project as well as slides for a talk I gave. IDnamegrade.csv contains a list of ID's linked to their corresponding classes. The full dataset can be obtained by contacting me at my website [website](mattschiffman.com).

## 3. src
This is where the bulk of the code is. Its broken into 3 main area - model, coinclasses, and scraper.
#### model
  * Main is a shortcut used to train/evaluate several of the models used in larger stages of my project. Its a good template but assumes the same filestructure I used for storing images.
  * TFModels. This is a class designed to train, evaluated, and test tensorflow models of coins. It take an encoding and a save folder to initialize. It allows you to train, evalauate, test, and predict models. It relies on coinlabel objects to index through the coins.
  * TFInput. These are helper functions for reading images in using tensorflow. The main function is read_input which takes a list of coin files created using the coin class. It can either read processed coisn (flatted binaries with the labels encoded in the last two bytes) or process them itself (much slower)
  * tf_helpers are helper functions for use in building up encodings for different graphs.  
  * model* are different models. They typically have a radial and "img" (standard) version to allow either image style to be used. model3c2d is currently the best performing model (it now only has two convolution layers but the original version had 3).
#### coinclasses
  * coinlabel. This class reads the csv of coins with their labels and allows for easy generation of file lists and train/test splits. The various model files had seeds in them to allow for consistent train/test splits.
  * coin. This class is for processing images. It takes an image and can crop, resize, and radially transform them. It allows coins to be saved as flattened array with encoded labels for later use.
  * make_coins. This is a file full of little scripts I wrote to do preprocessing to coins in bulk

#### scraper
  * make_url_list. Creates a csv file full of urls of ebay auctions for pcgs coins
  * Downloads all the coin images and ebay category information from a list of urls
