import en_core_web_sm
import pandas as pd
import spacy
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
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# from torch.utils.data import DataLoader, Dataset
from PIL import Image
# import numpy as np
# import torchvision.models as models
import torch.nn as nn
# import torch.optim as optim
# import urllib.request
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

nlp = en_core_web_sm.load()

nlp = spacy.load("en_core_web_sm")
emissions_csv = pd.read_csv(
    '/Users/aadrijupadya/Downloads/Food_Production.csv')


meta = [i for i in emissions_csv['Food product']]


def nlp_model(text_data):
    text_data = text_data.split(',')
    tokens = []
    for i in text_data:
        token = nlp(i)
        tokens.append(token)
    meta_tokens = []
    for food in meta:
        token = nlp(food)
        meta_tokens.append(token)
    print(len(tokens))
    print(len(meta_tokens))
    total_similarities = []
    for i in tokens:
        similarities = []
        for food in meta_tokens:
            token1, token2 = i, food
            similarities.append(token1.similarity(token2))
        total_similarities.append(similarities)
    max_items = [max(i) for i in total_similarities]
    print(len(max_items))
    max_indexes = []
    for i in total_similarities:
        max_indexes.append(i.index(max(i)))
    emissions = []

    res = {text_data[i]: max_indexes[i] for i in range(len(max_indexes))}
    for i in res:
        emissions.append(emissions_csv.iloc[res[i]]['Total_emissions'])
    final_string = f'Result: Estimate total emissions to be {round(sum(emissions), 2)} kg per 1 kg of this food'
    print(final_string)
    return final_string


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
        result = ""
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


@app.route('/inhome')
def inhome():
    return render_template("inhome.html")


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template("about.html")


@app.route("/post", methods=['GET', 'POST'])
def post():
    if "loggedin" in session:
        if request.method == "POST":
            recipe = request.form["recipe"]
            name = session["username"]
            cursor = mysql.connection.cursor()
            cursor.execute(
                ''' INSERT INTO recipes(name, recipe) VALUES(%s, %s) ''', (name, recipe))
            mysql.connection.commit()
            print("commited")
            return redirect(url_for("view"))
        return render_template("elements.html")
    else:
        return render_template("index.html")


@app.route("/text", methods=["GET", "POST"])
def text():
    if "loggedin" in session:
        if request.method == "POST":
            print("Post working")
            text_data = request.form['text']
            nlp_result = nlp_model(text_data)
            print(nlp_result)
            return render_template("text.html", nlp_result=nlp_result)
    else:
        return render_template("index.html")
    return render_template("text.html")


@app.route("/view", methods=["GET", 'POST'])
def view():
    if "loggedin" in session:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM recipes")
        recipes = cursor.fetchall()
        for rows in recipes:
            print(rows)
        return render_template("view.html", recipes=recipes)
    else:
        return render_template("index.html")


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
    return send_from_directory("uploads", filename)


@app.route("/logout")
def logout():
    session.pop("loggedin", None)
    session.pop("username", None)
    session.pop("registered", None)
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ''
    if request.method == "GET":
        if "loggedin" in session:
            return redirect(url_for('home'))
    elif request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT * FROM user_info WHERE gmail=%s AND password=%s", (email, password,))
        record = cursor.fetchone()
        try:
            rec_list = list(record.keys())
            val_list = list(record.values())
            ind = rec_list.index("first")
            user = val_list[ind]
            print(user)
        except:
            print("retry")
        if record:
            print(record)
            session["loggedin"] = True
            session["username"] = user
            return redirect(url_for("inhome"))
        else:
            print("Retry")
    return render_template("login.html", msg=msg)


if __name__ == "__main__":
    app.run(debug=True)
