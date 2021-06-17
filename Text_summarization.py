# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:18:27 2021

@author: Raju
"""
#This is just the import packages
import sys
import numpy as np # linear algebra
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import base64
import string
from collections import Counter
from time import time
import os
import nltk
from nltk.corpus import stopwords
import heapq
import warnings
warnings.filterwarnings('ignore')
stopwords = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
from os import listdir
import string
from pickle import dump,load

print("hello")

print("hi")
print("new change")
print("change from upstream")

class LoadData:
    def __init__(self, directory):
        self.directory= directory
        
    def load_stories(self):
        """
        Load the data and store it in a list of dictionaries
        """
        all_stories= list()
        
        def load_doc(filename):
            """
            Return the data from a given filename
            """
            file = open(filename, encoding='utf-8')
            text = file.read()
            file.close()
            return text
        
        def split_story(doc):
            """
            Split story from summaries based on the separater -> "@highlight"
            """
            index = doc.find('@highlight')
            story, highlights = doc[:index], doc[index:].split('@highlight')
            highlights = [h.strip() for h in highlights if len(h) > 0]
            return story, highlights
        
        list_of_files= listdir(self.directory)
        for name in list_of_files:
            filename = self.directory + '/' + name
            doc = load_doc(filename)
            story, highlights= split_story(doc)
            all_stories.append({'story': story, 'highlights': highlights})
        
        return all_stories
      
   



#stories_test[0]['story']
#stories_test[0]['story'].replace('\n','')

def preprocess(sentence):
        sentence=str(sentence)
        sentence = sentence.lower()
        sentence=sentence.replace('\n',"")
        rem_num = re.sub('[0-9]+', '', sentence)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)  
        filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
        stem_words=[stemmer.stem(w) for w in filtered_words]
        lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
        return " ".join(lemma_words)
#



    
    
# this is function for text summarization

def generate_summary(original_text, cleaned_text,file):
    sample_text = original_text
    doc = nlp(sample_text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(cleaned_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1 
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]


    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    file.write("Original Text::::::::::::\n")
    file.write(original_text)
    file.write('\n\nSummarized text::::::::\n')
    file.write(summary)
    file.write('\n\n')
  

def main():
    input_folder = sys.argv[1]
    outputfile ="output.txt"
    try:

        obj= LoadData( input_folder)
        stories_test= obj.load_stories()  
        clean_text=[]
        for i in range(len(stories_test)):
            clean_text.append(preprocess(stories_test[i]['story']))
        with open(outputfile, 'w',encoding='utf-8') as file:
            [generate_summary(stories_test[i]['story'],clean_text[i],file) for i in             range(len(clean_text))]      
    except Exception as e:
        print(e)
            


if __name__=="__main__":
    main()
    
 
