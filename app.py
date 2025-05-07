from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import sqlite3
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = 'Models\BestModel-02_R2.pth'  # Update this path to your model file
app.config['DATABASE'] = 'predictions.db'

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
def init_db():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# Define the model architecture
class MelanomaCNN(nn.Module):
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256*15*15, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.pool(nn.ReLU()(self.bn4(self.conv4(x))))
        x = x.view(-1, 256*15*15)
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.dropout(nn.ReLU()(self.fc2(x)))
        x = self.fc3(x)
        return x

# Load the model
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MelanomaCNN().to(device)
        model.load_state_dict(torch.load(app.config['MODEL_PATH'], map_location = device))
        model.eval()  # Set to evaluation mode
        return model    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()
if model is not None:
    device = next(model.parameters()).device
    print(f"Model loaded successfully on device: {device}")
else:
    print("Failed to load model")

# Define transforms for prediction
prediction_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7164, 0.5676, 0.5455], std=[0.2248, 0.2135, 0.2276])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
    img = Image.open(image_path)
    img = prediction_transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def process_image(image_path):
    """Process the image and return prediction results"""
    try:
        if model is None:
            return {'error': 'Model not loaded'}, 500

        # Get the device the model is on
        device = next(model.parameters()).device

        # Preprocess the image
        processed_image = preprocess_image(image_path)
        processed_image = processed_image.to(device)  # Move input to the same device as model
        
        # Make prediction
        with torch.no_grad():
            prediction = model(processed_image)
            probabilities = torch.softmax(prediction, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence = float(confidence[0])
            result = 'Benign' if predicted_class[0] == 0 else 'Malignant'
        
        # Save prediction to database
        with sqlite3.connect(app.config['DATABASE']) as conn:
            conn.execute(
                'INSERT INTO predictions (image_path, prediction, confidence) VALUES (?, ?, ?)',
                (image_path, result, confidence)
            )
            conn.commit()
        
        return {
            'prediction': result,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {'error': str(e)}, 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.execute('''
                SELECT id, image_path, prediction, confidence, timestamp 
                FROM predictions 
                ORDER BY timestamp DESC
            ''')
            predictions = []
            for row in cursor.fetchall():
                predictions.append({
                    'id': row[0],
                    'image_url': row[1],
                    'prediction': row[2],
                    'confidence': row[3],
                    'timestamp': row[4],
                    'recommendations': get_recommendations(row[2])
                })
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        print(f"Error loading history: {str(e)}")
        return render_template('history.html', predictions=[])

def get_recommendations(prediction):
    if prediction == 'Benign':
        return [
            'Continue regular skin checks',
            'Use sunscreen daily',
            'Monitor any changes in the lesion',
            'Schedule annual dermatologist visit'
        ]
    else:
        return [
            'Schedule an immediate appointment with a dermatologist',
            'Document the lesion with photos',
            'Avoid sun exposure to the area',
            'Follow up with your primary care physician'
        ]

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = process_image(filepath)
            if isinstance(result, tuple) and result[1] == 500:
                return jsonify(result[0]), 500
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    init_db()
    app.run(debug=True) 