from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import cv2
import os
from skimage.feature import hog
from skimage import exposure
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- LOAD MODEL ---
try:
    with open('model_svm_iklan.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: Model tidak ditemukan!")
    exit()

scan_history = []

def process_and_visualize(image_path, filename):
    IMG_SIZE = 128
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    fd, hog_image = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    
    # Meningkatkan kontras HOG agar terlihat keren di web
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) * 255
    hog_filename = 'hog_' + filename
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], hog_filename), hog_image_rescaled)
    return fd, hog_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        if 'imagefile' not in request.files: return redirect(request.url)
        file = request.files['imagefile']
        if file.filename == '': return redirect(request.url)

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                features, hog_filename = process_and_visualize(filepath, filename)
                pred_idx = model.predict([features])[0]
                
                timestamp = datetime.now().strftime("%H:%M")
                
                if pred_idx == 0: 
                    label = "IKLAN TERDETEKSI"
                    subtext = "Mengandung pola visual promosi."
                    theme = "danger" 
                    icon = "bi-exclamation-shield-fill" # Icon baru
                else:
                    label = "KONTEN AMAN"
                    subtext = "Tidak ditemukan indikasi iklan."
                    theme = "success" 
                    icon = "bi-check-circle-fill" # Icon baru

                # Simpan ke history
                scan_history.insert(0, {'time': timestamp, 'label': label, 'theme': theme})

                result = {
                    'label': label,
                    'subtext': subtext,
                    'theme': theme,
                    'icon': icon,
                    'img_ori': filepath,
                    'img_hog': os.path.join(app.config['UPLOAD_FOLDER'], hog_filename)
                }
                
            except Exception as e:
                print(e)

    return render_template('index.html', result=result, history=scan_history[:4])

if __name__ == '__main__':
    app.run(debug=True)