from flask import Flask, render_template, request, url_for, redirect, session, flash, send_from_directory
from flask_mysqldb import MySQL
from datetime import timedelta
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

import os
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import urllib.request
import pickle


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


device = set_device()


def pre_image(image_path, model):

    img = Image.open(image_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    # input = Variable(image_tensor)
    img_normalized = img_normalized.to(device)
    # print(img_normalized.shape)
    with torch.no_grad():
        model.eval()
        output = model(img_normalized)
       # print(output)
        index = output.data.cpu().numpy().argmax()
        classes = {0: 'Definitely Healthy', 1: 'Definitely Unhealthy',
                   2: 'Healthy', 3: 'Unhealthy'}
        m = nn.Softmax(dim=1)
        results = m(output)
        results = sorted(results)
        class_name = classes[index]
        result_name = f'{class_name}, Confidence Value: {round(torch.max(results[0]).item() * 100, 2)} %'
        print(result_name)
        return result_name


filename = 'resnet18_torch_trained_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


app = Flask(__name__)
app.secret_key = "appathon"
app.permanent_session_lifetime = timedelta(days=5)


app.config["MYSQL_HOST"] = 'localhost'
app.config["MYSQL_USER"] = "root"
app.config['MYSQL_PASSWORD'] = "Aadrij2005"
app.config['MYSQL_DB'] = "learn_users"
app.config['MYSQL_CURSORCLASS'] = "DictCursor"

mysql = MySQL(app)

app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)
res = []


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, "Only Images Are Allowed"),
            FileRequired("Field should not be empty")
        ]
    )
    submit = SubmitField("Upload")


@app.route('/', methods=["GET", "POST"])
def home():
    if "loggedin" in session:
        form = UploadForm()
        result = pre_image(
            '/Users/aadrijupadya/Downloads/cheetos.jpeg', loaded_model)
        if form.validate_on_submit():
            print("working")
            filename = photos.save(form.photo.data)
            result = pre_image('uploads/'+filename, loaded_model)
            print(result)
            file_url = url_for('get_file', filename=filename)
        else:
            file_url = None
            print("not working")
        return render_template("loggedin.html", username=session["username"], form=form, file_url=file_url, result=result)

    else:
        return render_template("index.html")
    return render_template("index.html")


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template("generic.html")


@app.route('/popreg')
def popreg():
    session.pop("registered", None)
    return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        if 'registered' in session:
            print("you've been redirected")
            return redirect(url_for("login"))
    elif request.method == "POST":
        first = request.form["first"]
        last = request.form["last"]
        email = request.form["email"]
        password = request.form["password"]
        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO user_info(first, last, gmail, password) VALUES(%s, %s, %s, %s) ''',
                       (first, last, email, password))
        mysql.connection.commit()
        session["registered"] = True
        print("in session")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/uploads/<filename>")
def get_file(filename):
    res.append(filename)
    if len(res) <= 1:
        print("list too small")
    else:
        os.remove("uploads/" + res[0])
        res.pop(0)
    print(res)
    return send_from_directory("uploads", filename)


@app.route("/logout")
def logout():
    session.pop("loggedin", None)
    session.pop("username", None)
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ''
    if request.method == "GET":
        if "loggedin" in session:
            return redirect(url_for('home'))
    elif request.method == "POST":
        if "registered" in session:
            email = request.form["email"]
            password = request.form["password"]
            cursor = mysql.connection.cursor()
            cursor.execute(
                "SELECT * FROM user_info WHERE gmail=%s AND password=%s", (email, password,))
            record = cursor.fetchone()
            rec_list = list(record.keys())
            val_list = list(record.values())
            ind = rec_list.index("first")
            user = val_list[ind]
            print(user)
            if record:
                print(record)
                session["loggedin"] = True
                session["username"] = user
                return redirect(url_for("home"))
            else:
                print("Retry")
        else:
            print("please register")
    return render_template("login.html", msg=msg)


if __name__ == "__main__":
    app.run(debug=True)
