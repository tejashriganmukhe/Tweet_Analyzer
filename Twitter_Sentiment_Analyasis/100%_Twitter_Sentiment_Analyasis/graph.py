
import tkinter as tk

from PIL import Image, ImageTk
from tkinter import ttk

import pandas as pd
import numpy as np
  
    
root = tk.Tk()
root.title("page")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
image2 =Image.open('b1.png')
image2 =image2.resize((w,h), Image.ANTIALIAS)


background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)
background_label.image = background_image

background_label.place(x=0, y=0)



lbl = tk.Label(root, text="___Twitter Analyzer System____", font=('Times New Roman', 35,' bold '),bg="black",fg="white",width=60,height=2)
lbl.place(x=0, y=0)



frame_alpr = tk.LabelFrame(root, text="Frame", width=300, height=600, bd=5, font=('times', 14, ' bold '),bg="#6495ED")
frame_alpr.grid(row=0, column=0)
frame_alpr.place(x=60, y=160)







def all_ana():
    
        
    
    frame_alpr1 = tk.LabelFrame(root, text="graph", width=800, height=600, bd=5, font=('times', 14, ' bold '),bg="#CDC9C9")
    frame_alpr1.grid(row=0, column=0)
    frame_alpr1.place(x=450, y=200)
    
    image2 =Image.open(r'D:/Python-Project-2022/Twitter_Sentiment_Analyasis/all.png')
    image2 =image2.resize((400,400), Image.ANTIALIAS)

    background_image=ImageTk.PhotoImage(image2)

    background_label = tk.Label(frame_alpr1, image=background_image)
    background_label.image = background_image

    background_label.place(x=100, y=50)

    
    

def pos():
    
        
    
    frame_alpr1 = tk.LabelFrame(root, text="graph", width=800, height=600, bd=5, font=('times', 14, ' bold '),bg="#CDC9C9")
    frame_alpr1.grid(row=0, column=0)
    frame_alpr1.place(x=450, y=200)
    
    image2 =Image.open(r'D:/Python-Project-2022/Twitter_Sentiment_Analyasis/positive.png')
    image2 =image2.resize((400,400), Image.ANTIALIAS)

    background_image=ImageTk.PhotoImage(image2)

    background_label = tk.Label(frame_alpr1, image=background_image)
    background_label.image = background_image

    background_label.place(x=100, y=50)
   

def neg():
    
        
    
    frame_alpr1 = tk.LabelFrame(root, text="graph", width=800, height=600, bd=5, font=('times', 14, ' bold '),bg="#CDC9C9")
    frame_alpr1.grid(row=0, column=0)
    frame_alpr1.place(x=450, y=200)
    
    image2 =Image.open(r'D:/Python-Project-2022/Twitter_Sentiment_Analyasis/negative.png')
    image2 =image2.resize((400,400), Image.ANTIALIAS)

    background_image=ImageTk.PhotoImage(image2)

    background_label = tk.Label(frame_alpr1, image=background_image)
    background_label.image = background_image

    background_label.place(x=100, y=50)

def neu():
    
        
    
    frame_alpr1 = tk.LabelFrame(root, text="graph", width=800, height=600, bd=5, font=('times', 14, ' bold '),bg="#CDC9C9")
    frame_alpr1.grid(row=0, column=0)
    frame_alpr1.place(x=450, y=200)
    
    image2 =Image.open(r'D:/Python-Project-2022/Twitter_Sentiment_Analyasis/negative.png')
    image2 =image2.resize((400,400), Image.ANTIALIAS)

    background_image=ImageTk.PhotoImage(image2)

    background_label = tk.Label(frame_alpr1, image=background_image)
    background_label.image = background_image

    background_label.place(x=100, y=50)
    


    
def window():
        root.destroy()
   
button1 = tk.Button(frame_alpr,command=all_ana,text="ALL Dataset Analysis",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button1.place(x=5,y=50)

button1 = tk.Button(frame_alpr,command=pos,text="Positive Analysis",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button1.place(x=5,y=150)


button1 = tk.Button(frame_alpr,command=neg,text="Negative Analysis",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button1.place(x=5,y=250)


button1 = tk.Button(frame_alpr,command=neu,text="Neutral Analysis",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button1.place(x=5,y=350)

# button2 = tk.Button(frame_alpr,command=Test,text="Test",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
# button2.place(x=5,y=140)

button3 = tk.Button(frame_alpr,text="Exit",command=window,bg="red",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button3.place(x=5,y=450)


root.mainloop()