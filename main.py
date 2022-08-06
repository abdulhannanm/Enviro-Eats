from flask import Flask, render_template, request, url_for, redirect, session, flash, send_from_directory
from flask_mysqldb import MySQL
from datetime import timedelta
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os




app = Flask(__name__)
app.secret_key = "appathon"
app.permanent_session_lifetime = timedelta(days=5)



app.config["MYSQL_HOST"] = 'localhost'
app.config["MYSQL_USER"] = "root"
app.config['MYSQL_PASSWORD'] = "ffrn1234"
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
        if form.validate_on_submit():
            print("working")
            filename = photos.save(form.photo.data)
            file_url = url_for('get_file', filename=filename)
        else:
            file_url = None
            print("not working")
        return render_template("loggedin.html", username = session["username"], form = form, file_url = file_url)
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
        cursor.execute(''' INSERT INTO user_info(first, last, gmail, password) VALUES(%s, %s, %s, %s) ''', (first, last, email, password))
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
            cursor.execute("SELECT * FROM user_info WHERE gmail=%s AND password=%s", (email, password,))
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
    return render_template("login.html", msg = msg)

if __name__ == "__main__":
    app.run(debug=True)