from flask import Flask, request, render_template
from func import *
import os

model =  r"E:\Users\HP\Desktop\aalix clg\projects\climate leaves\ayurmake\Ayurmake\model\model_sev.h5"
labels = r"E:\Users\HP\Desktop\aalix clg\projects\climate leaves\ayurmake\Ayurmake\model\labels.txt"
app = Flask(__name__)

@app.route("/", methods=["post", "get"])
def index():
    if request.method == 'POST':  
        f = request.files['file']
        path = f.filename
        f.save(path)
        re = process(path, model, labels)
        Vars = {
            "prediction":re["class"],
            "desc":desc(re["class"])
        }
        os.remove(path)
        return render_template("nagfani.html", **Vars)
    return render_template("index.html")


@app.route("/team")
def team():
    return render_template("team.html")


if __name__=="__main__":
    app.run(debug=True)