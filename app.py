import os
import shutil
from datetime import datetime

import pandas as pd
import requests # Still needed for potential future direct API calls, though verify_recaptcha is removed

import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.recaptcha import RecaptchaField
from werkzeug.security import check_password_hash, generate_password_hash
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import EqualTo, InputRequired, Length, ValidationError

# Local imports
from forecast import analyze_product_categories, create_seasonal_forecast
from pso_lstm import run_lstm_pso_forecast


# --- Flask Application Setup ---
app = Flask(__name__)

# --- Configuration ---
from flask_sqlalchemy import SQLAlchemy
import os

# Use environment variable or hardcoded URI
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL") or \
    "mysql+pymysql://flask_app_db_nlpk_user:d8Mg8IPdrKa9vOXiXFB7o5FTXd1PIhum@dpg-d1tl35mr433s73druqpg-a:5432/flask_app_db_nlpk"

# Add these to improve stability and avoid dropped connections:
app.config["SQLALCHEMY_POOL_RECYCLE"] = 280  # recycle connections before MySQL closes them
app.config["SQLALCHEMY_POOL_PRE_PING"] = True  # ping DB before using connection (avoids stale ones)

# Other config
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key_here"
app.config["RECAPTCHA_PUBLIC_KEY"] = "6LdYqocrAAAAAMKfjNKl4F24swM5tkWLK3f2x1rR"
app.config["RECAPTCHA_PRIVATE_KEY"] = "6LdYqocrAAAAANa485NBDEa1M-ASN9zcP-pE21PW"

# Initialize DB
db = SQLAlchemy(app)


# --- Database and Login Manager Initialization ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# --- File Paths ---
UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
BACKUP_FOLDER = "backups"
# LOG_FILE is no longer needed as logging is to DB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)


# --- Database Models ---
class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"User('{self.username}')"


class Log(db.Model):
    """Log model for application actions."""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.String(200))

    def __repr__(self):
        return f"Log('{self.timestamp}', '{self.action}', '{self.details}')"


# --- WTForms ---
class RegisterForm(FlaskForm):
    """Form for user registration."""
    username = StringField("Username", validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6, max=20)])
    confirm_password = PasswordField(
        "Confirm Password", validators=[InputRequired(), EqualTo("password", message="Passwords must match")]
    )
    submit = SubmitField("Register")

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError("That username is already taken. Please choose a different one.")


class LoginForm(FlaskForm):
    """Form for user login with reCAPTCHA."""
    username = StringField("Username", validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6, max=20)])
    recaptcha = RecaptchaField()
    submit = SubmitField("Login")


class ChangeUsernameForm(FlaskForm):
    """Form for changing username."""
    new_username = StringField("New Username", validators=[InputRequired(), Length(min=4, max=20)])
    submit = SubmitField("Change Username")

    def validate_new_username(self, new_username):
        if new_username.data != current_user.username:
            user = User.query.filter_by(username=new_username.data).first()
            if user:
                raise ValidationError("That username is already taken. Please choose a different one.")


# --- Helper Functions ---
@login_manager.user_loader
def load_user(user_id):
    """Loads a user from the database given their ID."""
    return User.query.get(int(user_id))


def log_to_database(action, details=""):
    """Logs an action to the database."""
    new_log = Log(action=action, details=details)
    db.session.add(new_log)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"ERROR - Failed to commit log entry: {e}")


# --- Context Processors ---
@app.context_processor
def inject_navigation():
    """Injects navigation links into all templates."""
    return dict(
        nav_links=[
            ("Upload", url_for("upload_page")),
            ("Filter", url_for("filter_sales")),
            ("Reports", url_for("reports")),
            ("Logs", url_for("view_logs")),
            ("Backup", url_for("backup")),
            ("Restore", url_for("restore")),
            ("Change Username", url_for("change_username")),
        ]
    )


# --- Routes ---
@app.route("/")
def landing():
    """Redirects to the login page."""
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handles user registration."""
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        try:
            db.session.commit()
            log_to_database("User Registration", f"User {form.username.data} registered.")
            flash(f"User {form.username.data} registered successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            flash(f"Error registering user: {e}", "danger")
    elif request.method == "POST":
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {getattr(form, field).label.text}: {error}", "danger")

    return render_template("register.html", form=form, site_key=app.config["RECAPTCHA_PUBLIC_KEY"])


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles user login."""
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            log_to_database("User Login", f"User {user.username} logged in.")
            flash("Logged in successfully!", "success")
            return redirect(url_for("upload_page"))
        else:
            flash("Invalid username or password.", "danger")
            log_to_database("Login Attempt Failed", f"Invalid login attempt for username: {form.username.data}")
    elif request.method == "POST":
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {getattr(form, field).label.text}: {error}", "danger")

    return render_template("login.html", form=form, site_key=app.config["RECAPTCHA_PUBLIC_KEY"])


@app.route("/users")
@login_required
def users():
    """Displays a list of all registered users."""
    users = User.query.all()
    return render_template("users.html", users=users)


@app.route("/logout")
@login_required
def logout():
    """Logs out the current user."""
    log_to_database("User Logout", f"User {current_user.username} logged out.")
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/back_to_upload")
@login_required
def back_to_upload():
    """Redirects back to the upload page."""
    return render_template("index.html", preserve_upload=True)


@app.route("/upload_page")
@login_required
def upload_page():
    """Renders the file upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    """Handles file upload and data processing."""
    if "file" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("upload_page"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file.", "danger")
        return redirect(url_for("upload_page"))

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        log_to_database("File Upload", f"User {current_user.username} uploaded file: {file.filename}")

        try:
            df = pd.read_csv(filepath, parse_dates=["transaction_date"])
            analyze_product_categories(df)
            forecast_results, arima_metrics = create_seasonal_forecast(df)
            log_to_database("Forecast Generation", "Generated ARIMA forecast.")
            lstm_metrics = run_lstm_pso_forecast(df.set_index("transaction_date")["quantity_sold"].resample("MS").sum())
            log_to_database("Forecast Generation", "Generated LSTM-PSO forecast.")

            preview_table = df.head().to_html(classes="table table-bordered")

            return render_template(
                "forecast_dashboard.html",
                plot1=url_for("static", filename="plots/category_seasonal_plot.png"),
                plot2=url_for("static", filename="plots/category_total_sales_plot.png"),
                plot3=url_for("static", filename="plots/final_seasonal_forecast_plot.png"),
                plot4=url_for("static", filename="plots/lstm_pso_forecast.png"),
                mape_arima=arima_metrics["MAPE"],
                rmse_arima=arima_metrics["RMSE"],
                r2_arima=arima_metrics["R2"],
                mape_lstm=lstm_metrics["MAPE"],
                rmse_lstm=lstm_metrics["RMSE"],
                r2_lstm=lstm_metrics["R2"],
                table_preview=preview_table,
            )
        except Exception as e:
            flash(f"Error processing file: {e}", "danger")
            log_to_database("File Processing Error", f"Error processing {file.filename}: {e}")
            return render_template("index.html", table_preview=None, error=str(e))
    flash("An unexpected error occurred during file upload.", "danger")
    return redirect(url_for("upload_page"))

@app.route('/filter', methods=['GET', 'POST'])
def filter_sales():
    uploaded_files = os.listdir(UPLOAD_FOLDER)

    if request.method == 'POST':
        selected_file = request.form.get('csv_file')
        filter_type = request.form.get('filter_type')

        if not selected_file:
            flash('Please select a CSV file.', 'danger')
            return redirect(url_for('filter_sales'))

        filepath = os.path.join(UPLOAD_FOLDER, selected_file)

        try:
            data = pd.read_csv(filepath)

            if filter_type == 'most_sold':
                filtered_data = data.sort_values(by='quantity_sold', ascending=False).head(10)
            elif filter_type == 'least_sold':
                filtered_data = data.sort_values(by='quantity_sold', ascending=True).head(10)
            else:
                flash('Invalid filter type!', 'danger')
                return redirect(url_for('filter_sales'))

            fig = px.bar(filtered_data, x='product_id', y='quantity_sold',
                         title=f"{filter_type.replace('_', ' ').title()} Products")
            fig.update_layout(
                xaxis_title='Product ID',
                yaxis_title='Quantity Sold'
            )
            graph_html = pio.to_html(fig, full_html=False)

            return render_template('forecast_dashboard.html',
                                   filtered_data=filtered_data.to_html(classes="table table-bordered"),
                                   graph_html=graph_html)

        except Exception as e:
            flash(f"Error processing file: {e}", "danger")

    return render_template('filter.html', files=uploaded_files)

@app.route("/reports")
@login_required
def reports():
    """Displays a list of uploaded files for reports."""
    files = os.listdir(UPLOAD_FOLDER)
    return render_template("reports.html", files=files)


@app.route("/logs")
@login_required
def view_logs():
    """Displays application logs from the database."""
    logs = Log.query.order_by(Log.timestamp.desc()).all()
    return render_template("logs.html", logs=logs)


@app.route("/backup")
@login_required
def backup():
    """Creates a zip backup of the uploaded files."""
    backup_filename = f"project_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    backup_path = os.path.join(BACKUP_FOLDER, backup_filename)
    shutil.make_archive(backup_path.replace(".zip", ""), "zip", UPLOAD_FOLDER)

    log_to_database("Data Backup", f"User {current_user.username} created a backup: {backup_filename}")
    flash("Backup created successfully!", "success")
    return send_file(backup_path, as_attachment=True)


@app.route("/restore", methods=["GET", "POST"])
@login_required
def restore():
    """Handles restoring files from a zip backup."""
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part.", "danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file.", "danger")
            return redirect(request.url)
        if file and file.filename.endswith(".zip"):
            try:
                # Clear existing files for a clean restore (optional but recommended)
                for f in os.listdir(UPLOAD_FOLDER):
                    file_path = os.path.join(UPLOAD_FOLDER, f)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                zip_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(zip_filepath)
                shutil.unpack_archive(zip_filepath, UPLOAD_FOLDER)
                os.remove(zip_filepath)

                log_to_database(
                    "Data Restore", f"User {current_user.username} restored data from {file.filename}."
                )
                flash("Files restored successfully!", "success")
                return redirect(url_for("reports"))
            except Exception as e:
                flash(f"Failed to restore files: {e}", "danger")
                log_to_database("Data Restore Failed", f"User {current_user.username} failed to restore: {e}")
        else:
            flash("Invalid file type. Please upload a .zip file.", "danger")

    return render_template("restore.html")


@app.route("/download/<filename>")
@login_required
def download(filename):
    """Allows downloading of uploaded files."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        flash("File not found.", "danger")
        return redirect(url_for("reports"))


@app.route("/change_username", methods=["GET", "POST"])
@login_required
def change_username():
    """Allows the current user to change their username."""
    form = ChangeUsernameForm()
    if form.validate_on_submit():
        old_username = current_user.username
        new_username = form.new_username.data

        if old_username == new_username:
            flash("New username is the same as your current username. No changes made.", "info")
            return redirect(url_for("change_username"))

        try:
            current_user.username = new_username
            db.session.commit()
            log_to_database("Username Change", f"Username changed from {old_username} to {new_username} for user ID {current_user.id}.")
            flash(f"Your username has been changed from {old_username} to {new_username}!", "success")
            return redirect(url_for("upload_page"))
        except Exception as e:
            db.session.rollback()
            log_to_database("Username Change Failed", f"Failed to change username for {old_username} to {new_username}: {e}")
            flash(f"Error changing username: {e}", "danger")
    elif request.method == "GET":
        form.new_username.data = current_user.username

    return render_template("change_username.html", form=form)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()