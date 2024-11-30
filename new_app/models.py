from django.db import models
from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models


import numpy as np
import pickle

import json
from PIL import Image


# Testing phase
rf = pickle.load(open(r"C:\Users\sarat\Music\Twitter Disaster\FINAL CODE\twitter_RF_bow.pkl", 'rb'))

countvect = pickle.load(open(r"C:\Users\sarat\Music\Twitter Disaster\FINAL CODE\count_vect.pkl", 'rb'))


def predict(text,algo): 
	text = [text]
	filter_text = countvect.transform(text)
	print(filter_text.shape)
	if algo=='rf':
		y_pred=rf.predict(filter_text)
		return y_pred[0]
	else:
		y_pred=rf.predict(filter_text)
		return y_pred[0]

