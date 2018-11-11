First step for training model is to generate the samples with generate_samples.py:
	./generate_samples.py [Path to save samples]
Note:
	You should install the python library needed of the project with pip before you generate the samples.

Second step for training model is to execute the train.py:
	./train.py [Samples Path] [Checkpoint Path for saving model] [ Checkpoint Path for restoring model]
Note:
	1. The new image samples must be resized into 64*64 and named by "-[0-9].jpg" in the samples images path.
	2. Output format of training process:
		step: [step index],loss:[loss value],train_acc:[Accuray of the samples used to train model],test_acc:[Accuray of the test samples used to evaluate the performance of the model]
	3. The model will be restored from the path if you specific argument [Checkpoint Path for restoring model]. Otherwise the model will be retrained and the checkpoint will be saved into [Checkpoint Path for saving model]
