# SL_Predictor
Authors: Hannah McDonald, Pranav Vempati

## Description:
A gene pair is considered synthetic lethal if the disturbance (e.g., knockdown, mutation, etc.) of <br>
both genes results in cell death meanwhile pertubation of only one of the genes does not impact cell <br>
viability. Synthetic lehality has implications in the research of cancer treatments as it indicates<br>
possible drug targets. However, determining which of the many combinations of genes form synthetic <br>
lethal(SL) pairs experimentally is time consuming.<br>

SL_Predictor is a feed forward fully-connected neural network that aims to speed up the process of <br>
identifying synthetic lethal gene pairs by predicting which genes are most likely to form synthetic <br>
lethal pairs based on their functional annotations.<br>

The model uses information about known SL and non-SL gene pairs provided in the SynLethDB database. <br>
This dataset contains data obtained through computational and biological methods, and for this model, <br>
only the data found experimentally was used. Additionally, the data contains information for a variety <br>
of cell-lines. Given research that suggests synthetic lehality is cell-line specific, only data <br>
corresponding to a single cell-line(K562: leukemia) were selected during data preparation.The SynLethDB <br>
data was cross-referrenced with the KEGG Brite database for the functional information of the genes. <br>
This data was prepared with Preprocessing.py and is stored in FunctionMapping.txt for use in model.py.<br>


### Model Architecture:
As mentioned previously, SL_Predictor is a feed forward fully-connected neural network. It uses 112 <br> 
input features corresponding to the KEGG Brite functional annotations for each gene. In other words,<br> 
the first 56 features is the one-hot encoding of the first gene's functions while the last 56 features<br> 
indicate that of the second gene in the pair. Through trial and error, a single hidden layer with a<br> 
width of 30 neurons was determined to result in the best performance. The model uses binary cross <br> 
entropy loss for training. In order to account for the severe class imbalance of the data, the class<br> 
weights were considered during loss calculation and AUROC scores were used as the performance metric. <br> 


### Tuning:
The Ray python library was used in order to perform tuning with grid search. The grid search was <br> 
configured to test the following range of hyperparameters: learning rate = [0.0005, 0.00075, 0.001, <br> 
0.0025, 0.005] and number of epochs = [25, 50, 75, 100, 125]. A weight decay of 0.0001 was used with<br> 
the Adam optimizer but this was not tuned due to the time requirements of grid search. Additionally,<br> 
K-Folds cross validation was used and the mean of the results was used as the tuning result for a <br> 
given set of hyperparameters during grid search.<br> 


### Results


## Requirements


## Usage Instructions


## Contact
Please email Hannah McDonald (mcdonaldhannah2000@gmail.com) with any questions.