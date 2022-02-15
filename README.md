# SL_Predictor
Authors: Hannah McDonald, Pranav Vempati

<h2>Description</h2>
<p>A gene pair is considered synthetic lethal if the disturbance (e.g., knockdown, mutation, etc.) of both genes<br> 
results in cell death while pertubation of only one of the genes does not impact cell viability. Synthetic<br> lehality has implications in the research of cancer treatments as it indicates possible drug targets.<br>
However, determining which of the many of combinations genes form synthetic lethal(SL) pairs experimentally is<br> time consuming.
</p>

<p>SL_Predictor is a neural network that aims to speed up the process of identifying synthetic lethal gene pairs<br> 
by predicting which genes are most likely to form synthetic lethal pairs based on their functional annotations.
</p>

<p>The model uses information about known SL and non-SL gene pairs provided in the SynLethDB database. This <br>dataset contains data obtained through computational and biological methods, and for this model, only the data<br> found experimentally was used.
Additionally, the data contains information for a variety of cell-lines. Given research<br> that suggests synthetic lehality
is cell-line specific, only data corresponding to a single cell-line(K562: leukemia)<br> were selected during data preparation.
The SynLethDB data was cross-referrenced with the KEGG Brite database<br> for functional information of the genes. This data was prepared with Preprocessing.py and is stored in<br> FunctionMapping.txt for use in model.py.

<h2>Requirements</h2>

<h2>Usage Instructions</h2>

<h2>Contact</h2>
<p>Please email Hannah McDonald (mcdonaldhannah2000@gmail.com) with any questions.</p>