from flask import Flask, render_template, request, send_file, url_for, redirect
import os, shutil
import pandas as pd
from forecast import analyze_product_categories, create_seasonal_forecast
from utils import calculate_metrics
from pso_lstm import run_lstm_pso_forecast
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
BACKUP_FOLDER = 'backups'
LOG_FILE = 'logs.txt'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)

# Context processor to inject nav links
@app.context_processor
def inject_navigation():
    return dict(
        nav_links=[
            ("Home", url_for('index')),
            ("Reports", url_for('reports')),
            ("Logs", url_for('view_logs')),
            ("Backup", url_for('backup')),
            ("Restore", url_for('restore'))
        ]
    )

def log_action(action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {action}\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    log_action(f"Uploaded file: {file.filename}")

    try:
        df = pd.read_csv(filepath, parse_dates=["transaction_date"])
        analyze_product_categories(df)
        forecast_results, arima_metrics = create_seasonal_forecast(df)
        log_action("Generated ARIMA forecast")
        lstm_metrics = run_lstm_pso_forecast(df.set_index('transaction_date')['quantity_sold'].resample('MS').sum())
        log_action("Generated LSTM-PSO forecast")
    except Exception as e:
        return f"Error processing file: {str(e)}"

    return render_template("forecast_dashboard.html",
        plot1=url_for('static', filename='plots/category_seasonal_plot.png'),
        plot2=url_for('static', filename='plots/category_total_sales_plot.png'),
        plot3=url_for('static', filename='plots/final_seasonal_forecast_plot.png'),
        plot4=url_for('static', filename='plots/lstm_pso_forecast.png'),
        mape_arima=arima_metrics["MAPE"], rmse_arima=arima_metrics["RMSE"], r2_arima=arima_metrics["R2"],
        mape_lstm=lstm_metrics["MAPE"], rmse_lstm=lstm_metrics["RMSE"], r2_lstm=lstm_metrics["R2"]
    )

@app.route("/reports")
def reports():
    files = os.listdir(UPLOAD_FOLDER)
    return render_template("reports.html", files=files)

@app.route("/logs")
def view_logs():
    with open(LOG_FILE, 'r') as f:
        log_content = f.readlines()
    return render_template("logs.html", logs=log_content)

@app.route("/backup")
def backup():
    backup_path = os.path.join(BACKUP_FOLDER, 'project_backup.zip')
    shutil.make_archive(backup_path.replace('.zip', ''), 'zip', UPLOAD_FOLDER)
    log_action("Downloaded backup")
    return send_file(backup_path, as_attachment=True)

@app.route("/restore", methods=["GET", "POST"])
def restore():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        log_action(f"Restored file: {file.filename}")
        return redirect('/reports')
    return render_template("restore.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
