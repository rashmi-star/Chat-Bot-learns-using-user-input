import random
import json
import pickle
import numpy as np
import subprocess


import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model #'from tensorflow.keras.models import load_model'is not working
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


    
def clean_up_sentence(sentence):
    lemmatizer=WordNetLemmatizer()
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):# sentence into a list full of 0 and 1 which means words present or not
    words=pickle.load(open('words.pkl','rb'))
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
         for i, word in enumerate(words):
             if word == w:
                 bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    classes=pickle.load(open('classes.pkl','rb'))
    model=load_model("chatbot_model.h5")

    bow= bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]#enumerate

    results.sort(key=lambda x:x[1], reverse=True)#learn lambda(probability)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def runpls():
    subprocess.run(["python","training.py"])


def get_response(intents_list, intents_json):
    tag= intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result
    
print("Lets start")

@app.route("/")
def index():
    return render_template('base.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    filename="intents.json"
    msg = request.form["msg"]
    input = msg
    a="I have learnt that"
    if input.startswith("!"):
        pls=input.split(":")
        tag=pls[0]
        patterns=pls[1].split(",")
        responses=pls[2].split(",")
        with open(filename, 'r') as json_file:
            listObj = json.load(json_file)
        if tag in listObj["intents"].__str__():
            o="Tag already present, please enter different tag"
            return o
        else:
            if "intents" in listObj:#change tag without fail
                listObj["intents"].append({
            "tag":tag,
            "patterns": patterns,
            "responses": responses
            }) 
                with open(filename, 'w') as json_file:
                    json.dump(listObj, json_file, 
                        indent=4,  
                        separators=(',',': '))
        subprocess.run(["python","training.py"])
        print(listObj)
        return a
    elif input=="learn":#try if else(remember it takes even one pattern and response)fina
            b="what should i learn?\n format- !tag:keywords:answers"
            return b            
    else:
        intents=json.loads(open('intents.json').read())
        ints=predict_class(input)
        res=get_response(ints,intents)
        return res
 
if __name__ == '__main__':
    app.run()








