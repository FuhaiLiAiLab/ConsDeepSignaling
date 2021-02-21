# ConsDeepSignaling

In this study, we proposed a not fully connected deep learning model, ConsDeepSignaling, for drug effect prediction. 

### Dependencies
* python 3.7.3
* tensorflow 1.13.1
* pandas
* sklearn

## 1.Data Preprocess
This study intergrates following datasets
* GDSC drug effect dataset 
* Gene expression data of GDSC
* KEGG signaling pathways and cellular process
* Drug-Target interactions from DrugBank database

GDSC data sets are included in folder *GDSC*, and other data sets are included in *init_data* and *mid_data*.  

In the main function of *parse_file.py*, we can adjust *k* to choose number of folds and choose *place_num* to get certain split of data set.  

Finally, those datasets files will be parsed into numpy files to train our ConsDeepSignaling model.  


```
python3 parse_file.py
python3 load_data.py
```

## 2.Running ConsDeepSignaling
Run the code
```
python3 main.py
```
Analyze the experiment results and plot figures
```
python3 analysis.py
```