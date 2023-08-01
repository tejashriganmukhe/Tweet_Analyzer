import pandas as pd, numpy as np, re

from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
import matplotlib.pyplot as plt

#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################
    
root = tk.Tk()
root.title("TWEET ANALYSER SYSTEM")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
#from tkvideo import tkvideo
# video_label =tk.Label(root)
# video_label.pack()
# read video to display on label
# player = tkvideo("acci.mp4", video_label,loop = 1, size = (w, h))
# player.play()
image2 =Image.open(r'img3.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)\
    
    
    

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

###########################################################################################################
lbl = tk.Label(root, text="___TWEET ANALYSER SYSTEM___", font=('times', 35,' bold '), height=1, width=60,bg="#FFBF40",fg="black")
lbl.place(x=0, y=0)
##############################################################################################################################


def Data_Display():
    columns = ['Headline', 'CLasses']
    print(columns)

    data1 = pd.read_csv(r"D:/Python-Project-2022/sentiment analysis/sentiment emotion/sentiment emotion based on twitter/sent (Autosaved).csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    #article_link = data1.iloc[:, 0]
    headline = data1.iloc[:, 0]
    classes = data1.iloc[:, 1]


    display = tk.LabelFrame(root, width=100, height=400, )
    display.place(x=270, y=100)

    tree = ttk.Treeview(display, columns=(
    'Headline', 'Classes'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=40)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Calibri', 10), background="black")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2")
    tree.column("1", width=130)
    tree.column("2", width=150)
    #tree.column("3", width=200)

    tree.heading("1", text="Headline")
    tree.heading("2", text="Classses")
    #tree.heading("3", text="Is_sarcastic")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 304):
        tree.insert("", 'end', values=(
        headline[i], classes[i]))
        i = i + 1
        print(i)

##############################################################################################################

def all_ana():
    
        
    
    # frame_alpr1 = tk.LabelFrame(root, text="graph", width=800, height=600, bd=5, font=('times', 14, ' bold '),bg="#CDC9C9")
    # frame_alpr1.grid(row=0, column=0)
    # frame_alpr1.place(x=450, y=200)
    
    image2 =Image.open(r'D:/Python-Project-2022/Twitter_Sentiment_Analyasis/graph.png')
    image2 =image2.resize((400,400), Image.ANTIALIAS)

    background_image=ImageTk.PhotoImage(image2)

    background_label = tk.Label(root, image=background_image)
    background_label.image = background_image

    background_label.place(x=850, y=150)

def Train():
    
    result = pd.read_csv("sent (Autosaved).csv",encoding = 'unicode_escape')

    result.head()
        
    result['headline_without_stopwords'] = result['headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
 ###########################################################################################################################################
    
    def pos(review_without_stopwords):
        return TextBlob(review_without_stopwords).tags
    
    
    os = result.headline_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()
    
    os1['pos'] = os1['headline_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    review_train, review_test, label_train, label_test = train_test_split(result['pos'], result['classes'],
                                                                              test_size=0.2,random_state=8)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(review_train)
    X_test_tf = tf_vect.transform(review_test)
    
    
    ###########################################################################################################################
    # def svc_param_selection(X, y, nfolds):
    #     Cs = [0.001, 0.01, 0.1, 1, 10]
    #     gammas = [0.001, 0.01, 0.1, 1]
    #     param_grid = {'C': Cs, 'gamma': gammas}
    #     grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    #     grid_search.fit(X, y)
    #     return grid_search.best_params_
    
    
    # svc_param_selection(X_train_tf, label_train, 5)
    #
    
    clf = svm.SVC(C=10, kernel='linear', gamma=0.001,random_state=123)   
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(review_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=960,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=960,y=300)
    
    dump (clf,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")
    
################################################################################################################################################################

frame = tk.LabelFrame(root,text="Control Panel",width=250,height=600,bd=3,background="cyan2",font=("Tempus Sanc ITC",15,"bold"))
frame.place(x=15,y=100)

entry = tk.Entry(frame,width=18,font=("Times new roman",15,"bold"))
entry.insert(0,"Enter text here...")
entry.place(x=25,y=150)
##############################################################################################################################################################################
def Test():
    predictor = load("SVM_MODEL.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
    if y_predict[0]==0:
        label4 = tk.Label(root,text ="Neutral Sentence",width=20,height=2,bg='#46C646',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    elif y_predict[0]==1:
        label4 = tk.Label(root,text ="Positive Sentence",width=20,height=2,bg='#46C646',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    else:
        label4 = tk.Label(root,text ="Negative Sentence",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
    
###########################################################################################################################################################
def graph():
   from subprocess import call
   call(["python", "graph.py"])
    
def window():
    root.destroy()
    

# button2 = tk.Button(frame,command=Train,text="Train",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
# button2.place(x=25,y=100)


button2 = tk.Button(frame,command=all_ana,text="Data Analysis Graph",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button2.place(x=25,y=280)

button2 = tk.Button(frame,command=Data_Display,text="Data Display",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button2.place(x=25,y=60)

button3 = tk.Button(frame,command=Test,text="Test",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=25,y=200)


button2 = tk.Button(frame,command=graph,text="Graphs",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button2.place(x=25,y=360)

button4 = tk.Button(frame,command=window,text="Exit",bg="red",fg="black",width=15,font=("Times New Roman",15,"bold"))
button4.place(x=25,y=480)




root.mainloop()