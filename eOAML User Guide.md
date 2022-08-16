User Guide

Extended OAML has been developed to be easily executed on the Command Line (Terminal). Users are not required to interact with python scripts. Basic and Ensemble methods of Extended OAML are run with the same command line arguments (parameters).The requirements to install our framework are listed in Appendix D.

After installation, change directory to the “gama” folder
 
A sample run for the Extended OAML basic can be run as follows:
 
#0 Python Script
-	This could either be Extended_basic.py or Extended_ensemble.py. It tells the computer what Extended OAML approach to run

#1 Data Stream (str)
-	Path to the data stream for stream learning. Data Streams must be arff files. If you would like to include a new data stream. Add the data stream to the gama/data_streams folder and include its path in gama/data_stream/data_utils.py.

#2 Batch size (int)
-	Initial number of data instances used in OAML search

#3 Sliding window size (int)
-	Number of data instances used in re-training OAML after a trigger point

#4 Gama evaluation metric (str)
-	Evaluation metric used in optimizing the search phase 
[acc, f1, roc_auc, b_acc] for classification
[rmse, mse] for regression
-	If the metric does not match the dataset, (e.g., mse should only be used for regression tasks), the OAML system throws an error

#5 River evaluation metric (str)
-	Evaluation metric for the online learning algorithm
[acc, f1, roc_auc, b_acc] for classification
[rmse, mse] for regression
-	If the metric does not match the dataset, (e.g., mse should only be used for regression tasks), the OAML system throws an error

#6 Time budget (int)
-	Time budget for the search algorithm

#7 Search Algorithm (str)
-	Optimisation algorithm used in the pipeline search
[random, evol]
 
#8 Live-Plot (Boolean)
-	Condition to create a live plot with Wandb-ai or not.
-	If True
-	Registration required on https://wandb.ai/site
-	Set entity to your wandb username 
-	Set project name as desired
-	[True, False]
