from flask import Flask, url_for, redirect, request, render_template, jsonify, session
from markupsafe import escape
import sqlite3


#jiaxing and jinxue
#ai chatbot implementation

#Xue Ping today
#finish diagnosis implementation

#minor things
#how to link jiaxing and jinxue models into our app and test (sync everything)
#figure out how to deploy flask on github
#understand favicons and the need to add them
#protect against injection attacks

app = Flask(__name__)
app.secret_key = "23d/fida6*dwk%$dz"

@app.route("/", methods = ["POST", "GET"])
def login():
    if request.method == "POST":
        Json = request.get_json()
        if valid_login(Json['username'], Json['password']):
            session['username'] = Json['username']
            return jsonify({"result": "success"})
        return jsonify({"result": "Incorrect password or username"})
    return render_template("login.html")

@app.route("/create account", methods = ["POST", "GET"])
def create_account():
    if request.method == "POST":
        Json = request.get_json()
        if unique_username(Json['username']):
            if Json['password'] == Json['confirm_password']:
                create_account_backend(Json['username'], Json['password'])
                return jsonify({"result": "Successful account creation"})
            return jsonify({"result": "Password and confirm password do not match!"})
        return jsonify({"result": "Username not unique!"})
    return render_template("createaccount.html")

@app.route("/successful login")
def proceed():
    if 'username' in session:
        return render_template("success.html")
    return redirect(url_for("login"))

@app.route("/homepage")
def homepage():
    if 'username' in session:
        return render_template("homepage.html")
    return redirect(url_for("login"))

@app.route("/report")
def report():
    if 'username' in session:
        return render_template("report.html")
    return redirect(url_for("login"))

@app.route("/process report", methods = ["POST"])
def process_report():
    if request.method == "POST":
        image_data = request.get_json()
        x = gen_ai_report(image_data)
        return jsonify({'data': x})

@app.route("/diagnosis")
def diagnosis():
    if 'username' in session:
        conn = sqlite3.connect('storage.db')
        c = conn.cursor()
        disease_row = tuple(c.execute('SELECT disease_name FROM diagnosis'))
        c.close()
        return render_template("diagnosis.html", diseases= disease_row)
    return redirect(url_for("login"))

@app.route("/diagnosis/<disease_name>")
def render_diagnosis(disease_name):
    if 'username' in session:
        conn = sqlite3.connect('storage.db')
        c = conn.cursor()
        query_row = tuple(c.execute('SELECT path FROM diagnosis WHERE disease_name =?', (disease_name,)))[0]
        c.close()
        return render_template(query_row[0])
    return redirect(url_for("login"))

@app.route("/chatterbot", methods = ["POST", "GET"])
def chatterbot():
    if 'username' in session:
        if request.method == "POST":
            return jsonify({"result": gen_ai_chatbot(request.get_json())})
        return render_template("chatterbot.html")
    return redirect(url_for("login"))

@app.route("/change password", methods = ["POST", "GET"])
def change_password():
    if 'username' in session:
        if request.method == "POST":
            Json = request.get_json()
            if valid_login(session['username'], Json['old_password']):
                if Json['password'] == Json['confirm_password']:
                    change_password_backend(session['username'], Json['password'])
                    return jsonify({"result": "Successful password change"})
                return jsonify({"result": "Password and confirm password do not match!"})
            return jsonify({"result": "Incorrect old password!"})
        return render_template("cp.html")
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for("login"))

def change_password_backend(username, new_password):
    conn = sqlite3.connect('storage.db')
    c = conn.cursor()
    insertTuple = (new_password, username)
    c.execute('UPDATE login SET password =? WHERE username =?', insertTuple)
    conn.commit()
    c.close()

def unique_username(username):
    conn = sqlite3.connect('storage.db')
    c = conn.cursor()
    usernameTuple = (username,)
    return_entry = tuple(c.execute('SELECT password FROM login WHERE username=?', usernameTuple))
    c.close()
    if len(return_entry) > 0:
        return False
    return True

def create_account_backend(username, password):
    conn = sqlite3.connect('storage.db')
    c = conn.cursor()
    insertTuple = (username, password)
    c.execute('INSERT INTO login (username, password) VALUES (?,?)', insertTuple)
    conn.commit()
    c.close()

def valid_login(username, password):
    conn = sqlite3.connect('storage.db')
    c = conn.cursor()
    usernameTuple = (username,)
    return_entry = tuple(c.execute('SELECT password FROM login WHERE username=?', usernameTuple))
    c.close()
    if len(return_entry) > 0:
        return_value = return_entry[0][0]
    else:
        return False
    return (return_value == password)

def gen_ai_report(image):
    import numpy as np
    import pytesseract
    import argparse
    import imutils
    import cv2

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    #make bytes to bytes 

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold the image using Otsu's thresholding method
    thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    # Set the path to the Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def image_to_text(image_path):
        image = Image.open(image_path)
        image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2)
        image = image.filter(ImageFilter.MedianFilter())

        custom_config = r'-l eng --oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return text

    new_image_path = thresh

    text_from_new_image = pytesseract.image_to_string(new_image_path)

    return text_from_new_image


def gen_ai_chatbot(text):
    return 'placeholder'