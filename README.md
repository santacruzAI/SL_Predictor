# SL_Predictor

Authors: Hannah McDonald, Pranav Vempati

## Description

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

### Model Architecture

As mentioned previously, SL_Predictor is a feed forward fully-connected neural network. It uses 112 <br> 
input features corresponding to the KEGG Brite functional annotations for each gene. In other words,<br> 
the first 56 features is the one-hot encoding of the first gene's functions while the last 56 features<br> 
indicate that of the second gene in the pair. Through trial and error, a single hidden layer with a<br> 
width of 30 neurons was determined to result in the best performance. The model uses binary cross <br> 
entropy loss for training. In order to account for the severe class imbalance of the data, the class<br> 
weights were considered during loss calculation and AUROC scores were used as the performance metric. <br> 

### Tuning

The Ray python library was used in order to perform tuning with grid search. The grid search was <br>
configured to test the following range of hyperparameters: learning rate = [0.0005, 0.00075, 0.001, <br> 
0.0025, 0.005] and number of epochs = [25, 50, 75, 100, 125]. A weight decay of 0.0001 was used with<br> 
the Adam optimizer.Additionally, K-Folds cross validation was utilized and the mean of the results was <br>
used as the tuning result for each round of the grid search. Through this process, the optimal <br> 
hyperparameter values were concluded to be a learning rate of 0.0025 and 25 epochs. <br> 

The weight decay value was not tuned using grid search because of the high time and memory requirements.<br>
Instead, the weight decay was tuned on its own after determining the learning rate and number of epochs<br>
via grid search. By iteratively training the model with a learning rate of 0.0025, 25 epochs, and <br> 
differing decay values, it was concluded that the optimal wweight decay value is 0.0075. <br> 

## Results

After tuning the model, it was trained once more using the determined hyperparameters (learning rate=0.0025,<br>
epochs=25, decay=0.0075). The training progress is shown in the figure below.<br>

![train_history.png](/train_history.png "Training progress over time.")<br>

<figcaption>Training progress over time.</figcaption>

Additionally, the final training and evaluation results are indicated in the figure below.<br>

![results.png](/results.png "Training and test results.")<br>

<figcaption>Training and evalauation results of the model.</figcaption>

With the high dimensionality of the features, it is difficult to visualize the correlations between <br>
the gene functions and the likelihood of pairs being predicted as SL or non-SL. In order to illustrate<br>
these relationships in a low dimensional space, a t-SNE plot was created as shown below.<br>

Because of the severe class imbalance of the data, the results shown in the t-SNE plot are somewhat<br>
speculative. However, as can be seen in the figure, there is some clear clustering of the SL pairs.<br>
This indicates that there is indeed a relationship between the associated gene functions and whether<br>
or not they display synthetic lethal interactions. This reinforces the results of the model evaluation<br>
described earlier.<br>

## Requirements

## Usage Instructions

## Next Steps

The work for this model is still ongoing and their are plans to expand upon it's usage. Primarily, we<br>
intend to improve upon the usablility of the program by organizing preprocessing.py and model.py in <br>
terms of functions to be called by a new program main.py that will, for example, allow users to train <br>
the model on other datasets or allow them to perform grid search with their own set of hyperparameters.<br>

Next, we will also create more visualizations of the data both to capture information about the data <br>
prior to training as well as additional figures depicting the model's performance in graphical form. <br>

Next, we hope to expand this model for use with other cell lines. To do this, we will use the same<br>
methods for deriving the data except we will select genes belonging to a different cell line. Then, <br>
we will use the pre-trained model and evaluate its performace on the new datasets. Additionally,<br>
although research suggests that many synthetic lethal pairings are cell-line specific, we may <br>
experiment with the model's performance on a dataset corresponding to more than one cell line.<br>

## Contact

Please email Hannah McDonald (mcdonaldhannah2000@gmail.com) with any questions.
