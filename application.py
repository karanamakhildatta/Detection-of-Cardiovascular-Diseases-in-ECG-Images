from tkinter import *
from tkinter.ttk import Combobox
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np





class ECG_Predictions:
    def __init__(self,root):
        self.root = root
        self.root.geometry("800x500+0+0")
        self.root.title("Cardiovascular Disease Prediction using Deeplearning")
        
        self.classes = ['Myocardial Infarction Patient','History of MI','abnormal heartbeat','Normal']

        self.model = load_model('./model_VGG19.h5')

        self.path = StringVar()

        label = Label(self.root,text="Cardiovascular Disease Prediction using Deep Learning",font=("times new roman",25,"bold"),bg="black",fg="gold",bd=4,relief=GROOVE)
        label.place(x=0,y=0,width=800,height=80)
        
        img = Image.open("main.jpg").resize((800,415))
        img = ImageTk.PhotoImage(img)
        label = Label(self.root,image=img)
        label.image = img
        label.place(x=0,y=85,width=800,height=415)


        btn = Button(self.root,text="Upload Image",font=('times new roman',16),bg='#1974d2',fg="#ffffff",command=self.upload)
        btn.place(x=25,y=440)

        btn = Button(self.root,text="Back",font=('times new roman',16),bg='#1974d2',fg="#ffffff",command=self.data_window)
        btn.place(x=200,y=440)

        btn = Button(self.root,text="Predict Image",font=('times new roman',16),bg='#1974d2',fg="#ffffff",command=self.predictttt)
        btn.place(x=645,y=440)



    def upload(self):
        path = askopenfilename()
        self.path.set(path)
    def data_window(self):
        self.root.destroy()
        self.screen = Tk()
        Heart_Disease(self.screen)
        self.screen.mainloop()
         
    def predictttt(self):
        k = self.path.get()
        if self.path.get() == "":
           return messagebox.showerror("Cardiovascular Disease Prediction","Please Upload a valid image")
        elif (k.split(".")[1] == "jpg") or (k.split(".")[1] == "png"):
            img=image.load_img(self.path.get(),target_size=(224,224))
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            a = self.model.predict(x)
            # print(a)
            p = self.classes[np.argmax(a)]
            messagebox.showinfo("Cardiovascular Disease Prediction",f"ECG of {p} person")
        else:
            messagebox.showerror("Cardiovascular Disease Prediction","Upload a valid image")


class Heart_Disease:
    def __init__(self,root):
        self.root = root
        self.root.geometry("500x500+0+0")
        self.root.title("Cardiovascular Disease Prediction using Deeplearning")
        self.frame = Frame(self.root,height=400)
        self.frame.place(x=0,y=90)
        
        label = Label(self.root,text="Cardiovascular Disease Prediction \n using Deep Learning",font=("times new roman",25,"bold"),bg="black",fg="gold",bd=4,relief=GROOVE)
        label.pack(fill=X)

        self.AGE = Label(self.frame,text="Enter Age",font=("times new roman",16,"bold"))
        self.AGE.grid(row=0,column=0)
        self.AGE_Entry = Entry(self.frame,width=21,font=("times new roman",16,"bold"),bd=4,relief=RIDGE)
        self.AGE_Entry.grid(row=0,column=1,padx=4)

        self.GENDER = Label(self.frame,text="Enter Gender",font=("times new roman",16,"bold"))
        self.GENDER.grid(row=1,column=0)
        self.GENDER_Entry = Combobox(self.frame,font=("times new roman",16,"bold"))
        self.GENDER_Entry['values'] = ["Male","Female"]
        self.GENDER_Entry['state'] = "readonly"
        self.GENDER_Entry.grid(row=1,column=1,padx=4,pady=4)

        self.WEIGHT = Label(self.frame,text="Enter Weight",font=("times new roman",16,"bold"))
        self.WEIGHT.grid(row=2,column=0)
        self.WEIGHT_Entry = Entry(self.frame,width=21,font=("times new roman",16,"bold"),bd=4,relief=RIDGE)
        self.WEIGHT_Entry.grid(row=2,column=1,padx=4,pady=4)

        self.BP = Label(self.frame,text="Enter BP",font=("times new roman",16,"bold"))
        self.BP.grid(row=3,column=0)
        self.BP_Entry = Entry(self.frame,width=21,font=("times new roman",16,"bold"),bd=4,relief=RIDGE)
        self.BP_Entry.grid(row=3,column=1,padx=4)


        self.SMOKING = Label(self.frame,text="Enter Smoking",font=("times new roman",16,"bold"))
        self.SMOKING.grid(row=4,column=0)
        self.SMOKING_Entry = Combobox(self.frame,font=("times new roman",16,"bold"),state="readonly")
        self.SMOKING_Entry['values'] = ("Yes","No")
        self.SMOKING_Entry.grid(row=4,column=1,padx=4,pady=4)

        self.ALCOHOL = Label(self.frame,text="Enter Alcohol",font=("times new roman",16,"bold"))
        self.ALCOHOL.grid(row=5,column=0)
        self.ALCOHOL_Entry = Combobox(self.frame,font=("times new roman",16,"bold"),state="readonly")
        self.ALCOHOL_Entry['values'] = ("Yes","No")
        self.ALCOHOL_Entry.grid(row=5,column=1,padx=4)

        self.PREDICT = Button(self.frame,text="Predict",font=("times new roman",16,"bold"),bg="black",fg="gold",bd=4,relief=RIDGE,command=self.predict)
        self.PREDICT.grid(row=6,column=0,pady=4)
        
        self.PREDICT = Button(self.frame,text="Predict using ECG",font=("times new roman",16,"bold"),bg="black",fg="gold",bd=4,relief=RIDGE,command=self.predict_ecg)
        self.PREDICT.grid(row=6,column=1,pady=4)

    def predict_ecg(self):
        self.root.destroy()
        self.screen = Tk()
        ECG_Predictions(self.screen)
        self.screen.mainloop()
    def predict(self):
        try:
            age = int(self.AGE_Entry.get())
            gender = self.GENDER_Entry.get()
            weight = int(self.WEIGHT_Entry.get())
            bp = self.BP_Entry.get()
            smoking = self.SMOKING_Entry.get()
            alcohol = self.ALCOHOL_Entry.get()
            if smoking == "Yes":
                smoking = 1
            else:
                smoking = 0
            if alcohol == "Yes":
                alcohol = 1
            else:
                alcohol = 0
            if gender == "Male":
                gender = 2
            else:
                gender = 1
        except:
            messagebox.showerror("Error","All Fields are required")
        if bp != "":
            try:
                bp = bp.split("/")
                systolic = int(bp[0])
                systolic = int(bp[1])
            except:
                messagebox.showerror("Error","Invalid BP")
                return
        model = pickle.load(open("model.pkl","rb"))
        k = [age,gender,weight,systolic,systolic,smoking,alcohol]
        k = np.array(k).reshape(1,-1)
        pred = int(model.predict(k)[0])
        print(pred)
        if pred == 0:
            messagebox.showinfo("Result","You are not at risk of Heart Disease")
        else:
            messagebox.showinfo("Result","You are at risk of Heart Disease\nECG is required to confirm the disease")
            self.root.destroy()
            self.screen = Tk()
            ECG_Predictions(self.screen)
            self.screen.mainloop()

            









if __name__  == "__main__":
    root = Tk()
    Heart_Disease(root)
    root.resizable(False,False)
    root.mainloop()