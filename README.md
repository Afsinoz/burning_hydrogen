We graciously acknowledge usage of data from the following sources:

### GEBCO - The General Bathymetric Chart of the Oceans
GEBCO Compilation Group (2024) GEBCO 2024 Grid (doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f)

### World Ocean Database
NOAA World Ocean Database (WOD) was accessed in July 2024 from https://registry.opendata.aws/noaa-wod

Reagan, James R.; Boyer, Tim P.; García, Hernán E.; Locarnini, Ricardo A.; Baranova, Olga K.; Bouchard, Courtney; Cross, Scott L.; Mishonov, Alexey V.; Paver, Christopher R.; Seidov, Dan; Wang, Zhankun; Dukhovskoy, Dmitry. (2024). World Ocean Atlas 2023. NOAA National Centers for Environmental Information. Dataset: NCEI Accession 0270533. Accessed July 2024.

### Aqua/MODIS (Moderate Resolution Imaging Spectroradiometer)
The Aqua/MODIS Chlorophyll-a Concentration dataset was acquired from the Ocean Biology (OB) Distributed Active Archive Center (DAAC), located in the Goddard
Space Flight Center in Greenbelt, Maryland (https://ladsweb.nascom.nasa.gov/).

### E.U. Copernicus Marine Service
Generated using E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00015, https://doi.org/10.48670/moi-00016

### COBE
COBE Sea Surface Temperature data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov 

COBE-SST 2 and Sea Ice data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov

Additionally, we graciously acknowledge PyTorch models modified from the following repositories: https://github.com/holmdk/Video-Prediction-using-PyTorch, https://github.com/ndrplz/ConvLSTM_pytorch

# Overview

In this project we examine ocean deoxygenation. The goal is to identify early indicators for ocean deoxygenation. Towards this goal, we trained models both to (1) predict oxygen concentration using other ocean variables and (2) predict oxygen concentration up to 60 days in the future using past data on oxygen and other ocean variables. Our models are equipped to handle data from the region from -100 degrees to 0 degrees latitude and -35 degrees latitude to 15 degrees latitude, gridded into 5 degree x 5 degree bins. The variables used in our models are: depth, temperature, salinity, chlorophyll-a concentration, O2 concentration, NO3 concentration, PO4 concentration, and Si concentration. The models can variously handle tabular data (pandas DataFrames) or array data (numpy arrays, PyTorch tensors). We discuss the folders in this repository below.

## data
This folder contains all of our data sources, stored as csv files.

## data_processing
This folder contains various files for downloading data from the internet, processing data into DataFrames, or merging several DataFrames together.

## numpy_dataset
This folder contains files for creating a single consolidated DataFrame and saving it as a csv file. It also contains files to transform this DataFrame into a 5-dimensional numpy array, normalize it, and impute missing values. Finally, the resulting csv files and numpy arrays are saved in this folder.

## models
This folder contains various models for studying ocean deoxygenation. Its sub-folders are described below.

### conv3d
The model in this folder predicts future concentrations of O2 in a 2-dimensional grid based on past values within the grid and the other variables in our dataset. This folder contains a PyTorch model consisting of a number of stacked Conv3d cells with skip connections in conv3d_classes.py. The model is trained on a gpu using gpu_training.ipynb. Model parameters stored during training are stored in the sub-folder saved_models. Some metrics evaluated during training are stored in the sub-folder runs. Finally, the sub-folder analysis contains some code for analyzing the errors of the model and visualizing its predictions. This model handles arrays of shape batch_size x num_channels x num_frames x height x width.

### convlstm
The model in this folder predicts future concentrations of O2 in a 2-dimensional grid based on past values within the grid and the other variables in our dataset. This folder contains a PyTorch model consisting of a number of stacked ConvLSTM cells in convlstm_classes.py. Images with 8 channels are fed into this model one at a time. The final resulting hidden states of the ConvLSTM cells are concatenated to form the output of the model. The model is trained on a gpu using gpu_training.ipynb. Model parameters stored during training are stored in the sub-folder saved_models. Some metrics evaluated during training are stored in the sub-folder runs. Finally, the sub-folder analysis contains some code for analyzing the errors of the model and visualizing its predictions. This model handles arrays of shape batch_size x num_frames x num_channels x height x width.

### encoder_decoder
The model in this folder predicts future concentrations of O2 in a 2-dimensional grid based on past values within the grid and the other variables in our dataset. This folder contains a PyTorch encoder-decoder model based on ConvLSTM cells in encoder_decoder_classes.py. Images with 8 channels are fed into the encoder one at a time. The decoder receives the hidden state of the encoder and continuously processes its own hidden final hidden state to produce as many new hidden states as desired. These hidden states are stacked and fed into a Conv3d cell to form the output of the model. The model is trained on a gpu using gpu_training.ipynb. Model parameters stored during training are stored in the sub-folder saved_models. Some metrics evaluated during training are stored in the sub-folder runs. Finally, the sub-folder analysis contains some code for analyzing the errors of the model and visualizing its predictions. This model handles arrays of shape batch_size x num_frames x num_channels x height x width.

### lstm
The model in this folder predicts future concentrations of O2 at a single location based on past concentrations of O2 at the location. This folder contains a PyTorch model based on stacked LSTM cells. Additionally the corresponding ipynb file performs some analysis of the errors of the model. This model handles arrays of shape batch_size x num_frames.

### no_model
The model in this folder predicts future concentrations of O2 in a 2-dimensional grid based on past values within the grid and the other variables in our dataset. This folder contains a baseline PyTorch model in no_model_classes.py which predicts that variables will stay constant in the future. This model handles arrays of shape batch_size x num_frames x num_channels x height x width.

### xgboost
The models in this folder predict oxygen concentration at a single location based on other variables measured at that location as well as their past values. Additionally, the ipynb files in this folder perform SHAP analysis to analyze the contributions of these various variables to the O2 concentration predicted by the model. These models handle tabular data.

### xgb_on_lin_reg
The model in this folder predicts future concentrations of O2 at a single location based on past variables at that location. The base model trains the an XGBoost model on the errors of a linear regression model. These base models can be chained using scikit-learn's RegressorChain class to produce multi-dimensional output. The sub-folder analysis performs some SHAP analysis on the model and analyzes its errors and visualizes its future predictions. This model handles tabular data.

### comparison
This folder contains various files for comparing the performance of the Conv3d, ConvLSTM, EncoderDecoderConvLSTM, and NoModel classes. Additionally it contains code to create gifs comparing the predictions of these models.
