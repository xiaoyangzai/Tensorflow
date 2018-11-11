Predicting the test samples by using file predict_model.py to load the pre-trained models file.
The usage of the predict_models.py:
	./predict_models.py [sample images path] [model checkpoint meta path] [model checkpoint path]
Note:
	1. The new image samples must be resized into 64*64 and named by "-[0-9].jpg" in the samples images path.
	2. output format: 
		Accuracy for all samples.
		[label of image the model predicted ] : [probability]
		[label of image the model predicted ] : [probability]
		...
		[label of image the model predicted ] : [probability]


predicting with the pre-trained model of digits:
	./predict_models.py ./digits_dataset_for_test ./classifier_digits/checkpoint/model_checkpoint.pkt.meta  ./classifier_digits/checkpoint


predicting with the pre-trained model of Cat.Vs.Dog:
	./predict_models.py ./catdog_dataset_for_test ./classifier_cat_dog/checkpoint/model_checkpoint.pkt.meta  ./classifier_cat_dog/checkpoint
