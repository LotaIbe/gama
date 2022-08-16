# Extended-OAML
Extension of the Online Automated Machine Learning Framework



The necessary files for installation are provided in the folder “Extended_OAML.zip”. Unzip the folder and perform the following steps. The unzipped folder consists of the following:
- gama
- river

1.	We recommend creating a python virtual environment before installation
2.	To install gama, change directory the gama folder and run the following
```python setup.py install```
3.	To install river, change directory to the river folder and run
```python setup.py install```
4. To install wandb, change directory to the river folder and run
```pip install wandb==0.9.7```
4.	It is important that you install river, and gama according to steps 2 and 3 above, to prevent version errors but also because we have modified these libraries for our system


If you use our applications, following are the user parameters and possible values. All 3 versions of OAML use the same
user parameters.
Alternatively, you can use GAMA with "online_learning=True" with your own script.

Example run:
    python OAML_basic.py 'data_streams/electricity-normalized.arff' 5000 5000 acc acc 60 evol False

#1 Data stream (str)
- Location of the stream you want to use for online learning. You can find example streams under "data_streams" folder.
- Must be arff file.

#2 Initial batch size (int)
- Number of data samples to be used for initial pipeline search.

#3 Sliding window size (int)
- Number of data samples to be used for pipeline update search at drift points.

#4 GAMA metric (str)
- Performance metric to optimize in GAMA pipeline search
- Should be one of the following:
     [acc, b_acc, f1, roc_auc, rmse, mse]

#5 River metric (str)
- Performance metric to evalue in Online learning.
- Should be one of the following:
    [acc, b_acc, f1, roc_auc, rmse, mse]

#6 Time budget (int)
- Time budget given to GAMA pipeline search.

#7 Search algorithm (str)
- Search algorithm to be used in GAMA pipeline search.
- Should be one of the following:
    [random, evol]

#8 Live-plot (Boolean)
- Whether to create a live plot with Wandb-ai or not.
- Requires registration to https://wandb.ai/site
- If True, then remember to change entity and project names in the script.
You can then follow your runs through Wand-ai under those pages.
- Should be one of the following:
    [True, False]


