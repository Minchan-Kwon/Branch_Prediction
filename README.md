# Branch_Prediction
This repository contains the necessary source code to collect data, train and test a neural branch predictor. 
For this project, the open-source PARSEC benchmark suite has been used. However, this choice is arbitrary and 
any benchmark suite that can be run on a Linux environment will suffice.

## Getting Started 

### Installing Tools
The project pipeline starts by collecting branch data that the predictor can train and test on. 
To collect data from a benchmark execution, the user should first download Intel Pin - a binary instrumentation tool letting the user analyze the execution of a program. 
You can download the toolkit [here](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html).
Ensure that the working Linux environment is set up for C++ development. 

Details on how to set up the Pin tool is written [here](./tools/README.md)
```

```
Once you clone the repository to your local machine, move "sampler.cpp" and "Makefile" located under "tools" to a new directory 

Installing parsec benchmark

Install Python Requirements 
```
python3 -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt
```


### Executing the Pipeline

1. predict_baseline

Using a 2-bit saturating counter as the baseline model, it will first print stats of the global branch history, and then make predictions on the branch history and save its predictions to the ../data directory. 
Look at the predictions and choose the appropriate PCs the model will train on. For this project, I have used 0x and 0x and ...

2. extract_branch_history

Using the global branch history csv file, it will create training data and its labels with respect to the given pc. 

3. Train

The user should Tell the directory in which the history, target, metadata are stored
Data is split into train, val, test and dataloader is created. 