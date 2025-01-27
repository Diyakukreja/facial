from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
import os
import shutil
from ultralytics import YOLO
import re
import werkzeug

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the skin recognition model (replace with your model's path)
model = YOLO('models/skin_recognition_model.pt')  # Use the path to skin_recognition_model.pt


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        # Sanitize filename
        filename = werkzeug.utils.secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure 'uploads' folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file.save(filepath)

        # Run YOLO prediction and save the output
        results = model.predict(source=filepath, conf=0.25, save=True)  # Save predictions to disk

        # Check the latest output directory created by YOLO
        output_parent_dir = 'runs/detect'
        output_dirs = [d for d in os.listdir(output_parent_dir) if d.startswith('predict')]

        # Filter out invalid directories (directories without numeric suffix)
        valid_dirs = [d for d in output_dirs if re.match(r'predict(\d+)', d)]

        if not valid_dirs:
            return "Error: No valid output directories found."

        # Find the most recent directory (based on numeric suffix)
        latest_dir = max(valid_dirs, key=lambda d: int(re.search(r'predict(\d+)', d).group(1)))
        output_dir = os.path.join(output_parent_dir, latest_dir)

        output_files = os.listdir(output_dir)
        if output_files:
            # Assuming the output image is saved with the same name as the input
            output_file = os.path.join(output_dir, output_files[0])  # First result in the output directory

            # Move the output to the desired location
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
            shutil.move(output_file, output_path)  # Move the file

            # Extract the labels (predictions)
            labels = results[0].names  # The class labels YOLO predicts
            if len(results[0].boxes) > 0:
                # Assuming we are interested in the first detected object
                predicted_label = labels[int(results[0].boxes[0].cls.item())]  # Extract label from the first detected box
            else:
                predicted_label = "No object detected"

            return render_template('result.html', output_image='output.jpg', label=predicted_label)
        else:
            return f"Error: No output image generated. Files found: {output_files}"


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
