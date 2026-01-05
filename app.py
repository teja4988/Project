import pandas as pd
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
import bcrypt
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from werkzeug.utils import secure_filename

# ------------------- Flask App Setup -------------------
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")  # MongoDB Atlas URI from .env

# ------------------- MongoDB Atlas Setup -------------------
client = MongoClient(MONGO_URI)
db = client['user_database']
collection = db['users']

# ------------------- Routes -------------------

@app.route('/')
def index():
    return render_template('index.html')

# ------------------- Registration -------------------
@app.route('/register')
def register_page():
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']

    # Hash password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Check duplicates
    if collection.find_one({'username': username}):
        session['error'] = "Username already exists!"
        return redirect(url_for('register_page'))
    elif collection.find_one({'email': email}):
        session['error'] = "Email already exists!"
        return redirect(url_for('register_page'))

    # Insert user
    user_data = {
        'username': username,
        'password': hashed_password,
        'email': email
    }
    collection.insert_one(user_data)
    return redirect(url_for('login_page'))

# ------------------- Login -------------------
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Dummy login
    if username == 'dummy' and password == 'dummy':
        session['username'] = 'Demo User'
        return redirect(url_for('dashboard', username='Demo User'))

    # MongoDB login
    user = collection.find_one({'username': username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        session['username'] = user['username']
        return redirect(url_for('dashboard', username=user['username']))
    else:
        session['error'] = "Invalid username or password"
        return redirect(url_for('login_page'))

# ------------------- Dashboard -------------------
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login_page'))

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login_page'))

@app.route('/admin')
def admin_page():
    if 'username' in session:
        return render_template('admin.html', username=session['username'])
    return redirect(url_for('login_page'))

# ------------------- Profile -------------------
@app.route('/profile')
def profile_page():
    if 'username' in session:
        user_data = collection.find_one({'username': session['username']})
        if user_data:
            return render_template('profile.html', username=session['username'], user_data=user_data)
        return "User not found in database"
    return redirect(url_for('login_page'))

@app.route('/profile/update', methods=['POST'])
def update_profile():
    if 'username' in session:
        username = session['username']
        updated_data = {
            'first_name': request.form.get('first_name'),
            'last_name': request.form.get('last_name'),
            'organization_name': request.form.get('organization_name'),
            'location': request.form.get('location'),
            'phone_number': request.form.get('phone_number'),
            'birthday': request.form.get('birthday')
        }
        collection.update_one({'username': username}, {'$set': updated_data})
        return redirect(url_for('profile_page'))
    return redirect(url_for('login_page'))

# ------------------- Fraud Prediction -------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    data = pd.read_csv(file_path)
    statistical_analysis = data.describe()
    fraudulent_count = (data['Class'] == 1).sum()
    non_fraudulent_count = (data['Class'] == 0).sum()

    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Isolation Forest
    iso_forest = IsolationForest()
    iso_forest.fit(X)
    iso_preds = iso_forest.predict(X)
    iso_preds = [-1 if p == -1 else 0 for p in iso_preds]
    iso_acc = accuracy_score(y, iso_preds)
    iso_err = 1 - iso_acc
    iso_report = classification_report(y, iso_preds)

    # SVM
    svm_model = SVC()
    svm_model.fit(X, y)
    svm_preds = svm_model.predict(X)
    svm_acc = accuracy_score(y, svm_preds)
    svm_err = 1 - svm_acc
    svm_report = classification_report(y, svm_preds)

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X, y)
    log_preds = logistic_model.predict(X)
    log_acc = accuracy_score(y, log_preds)
    log_err = 1 - log_acc
    log_report = classification_report(y, log_preds)

    return render_template(
        'admin.html',
        username=session['username'],
        statistical_analysis=statistical_analysis,
        fraudulent_count=fraudulent_count,
        non_fraudulent_count=non_fraudulent_count,
        iso_forest_accuracy=iso_acc,
        iso_forest_error=iso_err,
        iso_forest_classification_report=iso_report,
        svm_accuracy=svm_acc,
        svm_error=svm_err,
        svm_classification_report=svm_report,
        logistic_accuracy=log_acc,
        logistic_error=log_err,
        logistic_classification_report=log_report
    )

# ------------------- Logout -------------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

# ------------------- Main -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))