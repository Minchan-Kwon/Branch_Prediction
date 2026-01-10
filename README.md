# Branch_Prediction
This repository contains the necessary source code to collect data, train and test a neural branch predictor. 
For this project, the open-source PARSEC benchmark suite has been used. However, this choice is arbitrary and 
any benchmark suite that can be run in a Linux environment will suffice.

## Getting Started 

### Installing Tools
This project pipeline starts by collecting branch data that the predictor can train and test on. 
To collect data from a benchmark execution, the user should first download Intel Pin - a binary instrumentation tool letting the user analyze the execution of a program. 
You can download the toolkit [here](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html).
Ensure that the working Linux environment is set up for C++ development. 

Details on how to set up the Pin tool is written [here](./tools/README.md)
'''

'''
Once you clone the repository to your local machine, move "sampler.cpp" and "Makefile" located under "tools" to a new directory 