from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
file=open('model.pk1','rb')
clf=pickle.load(file)
file.close()

@app.route("/",methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        mydict=request.form
        Age=int(mydict['Age'])
        Diabetes=int(mydict['Diabetes'])
        Chd=int(mydict['Chd'])
        Cancer=int(mydict['Cancer'])
        Temprature=int(mydict['Temprature'])
        Pulse=int(mydict['Pulse'])
        Sys=int(mydict['Sys'])
        Dia=int(mydict['dia'])
        Rr=int(mydict['Rr'])
        Sats=int(mydict['Sats'])
        Cough=int(mydict['Cough'])
        Fever=int(mydict['Fever'])
        Diarrhea=int(mydict['Diarrhea'])
        Headache=int(mydict['Headache'])
        Loss_of_smell=int(mydict['Loss_of_smell'])
        Loss_of_taste=int(mydict['Loss_of_taste'])
        Runny_nose=int(mydict['Runny_nose'])
        Muscle_sore=int(mydict['Muscle_sore'])
        Sore_throat=int(mydict['Sore_throat'])
        #
        inputfeatures=[Age,Diabetes,Chd,Cancer,Temprature,Pulse,Sys,Dia,Rr,Sats,Cough,Fever,Diarrhea,Headache,Loss_of_smell,Loss_of_taste,Runny_nose,Muscle_sore,Sore_throat]
        infprob=clf.predict_proba([inputfeatures])[0][1]   
        print(infprob)
#'age', 'diabetes', 'chd', 'cancer', 'temperature', 'pulse', 'sys',
#'dia', 'rr', 'sats', 'cough', 'fever', 'diarrhea', 'headache',
#'loss_of_smell', 'loss_of_taste', 'runny_nose', 'muscle_sore',
#'sore_throat'    
        return render_template('show.html',inf=round(infprob*100))
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)   


#'age', 'diabetes', 'chd', 'cancer', 'temperature', 'pulse', 'sys',
#'dia', 'rr', 'sats', 'cough', 'fever', 'diarrhea', 'headache',
#'loss_of_smell', 'loss_of_taste', 'runny_nose', 'muscle_sore',
#'sore_throat'    