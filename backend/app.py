from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from predict import predict
from report import generate_pdf_report
import sqlite3
import os
import base64
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), 'drishti.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            patient_name TEXT,
            role TEXT,
            hospital TEXT,
            disease TEXT,
            confidence REAL,
            severity TEXT,
            recommendation TEXT,
            heatmap TEXT,
            original_image TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "DrishtiAI backend is running!"})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    patient_name = request.form.get('patient_name', 'Unknown')
    role = request.form.get('role', 'Unknown')
    hospital = request.form.get('hospital', 'Unknown')

    result = predict(image_bytes)
    original_b64 = base64.b64encode(image_bytes).decode('utf-8')

    report_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO reports (id, patient_name, role, hospital, disease, confidence,
                              severity, recommendation, heatmap, original_image, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (report_id, patient_name, role, hospital, result['disease'], result['confidence'],
          result['severity'], result['recommendation'], result['heatmap'], original_b64, created_at))
    conn.commit()
    conn.close()

    result['report_id'] = report_id
    result['created_at'] = created_at
    return jsonify(result)

@app.route('/history', methods=['GET'])
def history():
    patient_name = request.args.get('patient_name', '')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, patient_name, disease, confidence, severity, created_at
        FROM reports WHERE patient_name = ?
        ORDER BY created_at DESC
    ''', (patient_name,))
    rows = c.fetchall()
    conn.close()

    history_list = []
    for row in rows:
        history_list.append({
            'id': row[0],
            'patient_name': row[1],
            'disease': row[2],
            'confidence': row[3],
            'severity': row[4],
            'created_at': row[5]
        })
    return jsonify(history_list)

@app.route('/report/<report_id>', methods=['GET'])
def get_report(report_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Report not found'}), 404

    return jsonify({
        'id': row[0], 'patient_name': row[1], 'role': row[2], 'hospital': row[3],
        'disease': row[4], 'confidence': row[5], 'severity': row[6],
        'recommendation': row[7], 'heatmap': row[8], 'original_image': row[9],
        'created_at': row[10]
    })

@app.route('/download-pdf/<report_id>', methods=['GET'])
def download_pdf(report_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Report not found'}), 404

    report_data = {
        'id': row[0], 'patient_name': row[1], 'role': row[2], 'hospital': row[3],
        'disease': row[4], 'confidence': row[5], 'severity': row[6],
        'recommendation': row[7], 'heatmap': row[8], 'original_image': row[9],
        'created_at': row[10]
    }

    pdf_path = generate_pdf_report(report_data)
    return send_file(pdf_path, as_attachment=True,
                      download_name=f"DrishtiAI_Report_{report_data['patient_name'].replace(' ', '_')}.pdf")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)