# MSA Feature Extraction
MSA_feature_Extraction based on https://github.com/thuiar/MMSA-FET/tree/master/src/MSA_FET
## Preparation
Firstly, we should https://github.com/thuiar/MMSA-FET/tree/master/src/MSA_FET Install extraction package,and then use our files to start extracting.
## Extracting
The configs folder are configured methods for extracting information.
First, extract a single feature file for each sample using extractfeature.py.

Afterwards, use nofeat_mosi. py to save the existing information as a file to be supplemented, which contains information such as ID and source text, but the feature bar has no content.

Finally, mosi_pkl. py is responsible for saving all individual sample features into nofeat. pkl and merging them into a distinctive pkl file.
