from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import torch
import re 
from string import punctuation



app = Flask(__name__)

@app.route('/', methods=['POST','GET'])

def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]  
    return features

def tokenize_review(test_review):
    test_review = test_review.lower() 
    test_text = ''.join([c for c in test_review if c not in punctuation])
    test_words = test_text.split()
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])
    return test_ints

def predict(net, test_review, sequence_length=200): 
    net.eval()
    test_ints = tokenize_review(test_review)
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)  
    batch_size = feature_tensor.size(0)
    h = net.init_hidden(batch_size)
    
    output, h = net(feature_tensor, h)
    

    pred = torch.round(output.squeeze()) 
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")

def main():
	net = joblib.load('net.pkl')
	if request.method == 'GET':
		return render_template('index.html')

	if request.method == 'POST':
		review = request.form['review']	
		net = joblib.load('net.pkl')
		predict(net, review, seq_length)
		
if __name__ == "__main__":
    app.run()
