# ip_meld_estimation
Estimation of liver function by classifying the MELD-score with a CNN in contrast-enhanced MRI

requirements:
	python 3
	numpy (tested version: 1.15)
	matplotlib (tested version: 3.0)
	scikit-learn (tested version: 0.20)
	tensorflow (tested version: 1.10)
	

instructions for own dataset:
	1: create list of livers and their MELD-scores
	2: use 'create_Datafolder.py' to create class folders
	3: use 'get_MaxSize.py' to get max size for cropping
	4: use 'to_Npy.py' to save data in numpy arrays
	5: use 'shuffle_and_normalize.py' to normalize and shuffle data
	6: use 'training_and_prediction.py' to train and test a cnn
