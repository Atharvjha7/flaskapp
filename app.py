import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import Flask

app = Flask(__name__)
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageOps, ImageFilter
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure file upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure image display settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Import Flask-Login for user authentication
login_manager = LoginManager()
login_manager.init_app(app)

# Mock user data (replace with a real database)
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Initialize a list to store uploaded images
uploaded_images = []

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        uploaded_images.append(filename)
        flash('File uploaded successfully')
        return redirect(url_for('index'))
    else:
        flash('Invalid file format. Allowed formats: jpg, jpeg, png')
        return redirect(request.url)

@app.route('/images')
@login_required
def list_images():
    return render_template('list_images.html', images=uploaded_images)

@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare_images():
    if request.method == 'POST':
        image1 = request.form['image1']
        image2 = request.form['image2']

        if image1 not in uploaded_images or image2 not in uploaded_images:
            flash('Invalid image selections')
        else:
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'difference.jpg')
            if image_difference(os.path.join(app.config['UPLOAD_FOLDER'], image1),
                                os.path.join(app.config['UPLOAD_FOLDER'], image2),
                                output_path):
                flash('Images compared successfully')
                return redirect(url_for('list_images', filename='difference.jpg'))
            else:
                flash('Images are identical')

    return render_template('compare_images.html', images=uploaded_images)

@app.route('/annotate', methods=['GET', 'POST'])
@login_required
def annotate_images():
    if request.method == 'POST':
        image_to_annotate = request.form['image_to_annotate']
        annotation = request.form['annotation']

        if image_to_annotate in uploaded_images:
            # Implement image annotation logic here
            # You can save annotations in a database or in image metadata
            flash(f'Annotation added to {image_to_annotate}: {annotation}')
        else:
            flash('Invalid image selection')

    return render_template('annotate_images.html', images=uploaded_images)

@app.route('/filter', methods=['GET', 'POST'])
@login_required
def apply_filter():
    if request.method == 'POST':
        image_to_filter = request.form['image_to_filter']
        selected_filter = request.form['selected_filter']

        if image_to_filter in uploaded_images:
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered.jpg')
            apply_image_filter(os.path.join(app.config['UPLOAD_FOLDER'], image_to_filter),
                               selected_filter,
                               output_path)
            flash(f'Filter applied to {image_to_filter}')
            return redirect(url_for('list_images', filename='filtered.jpg'))
        else:
            flash('Invalid image selection')

    return render_template('apply_filter.html', images=uploaded_images)

@app.route('/export', methods=['GET', 'POST'])
@login_required
def export_images():
    if request.method == 'POST':
        images_to_export = request.form.getlist('images_to_export')
        if len(images_to_export) > 0:
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'exported_images.pdf')
            export_images_to_pdf([os.path.join(app.config['UPLOAD_FOLDER'], img) for img in images_to_export], pdf_path)
            flash('Images exported to PDF')
            return redirect(url_for('list_images', filename='exported_images.pdf'))
        else:
            flash('No images selected for export')

    return render_template('export_images.html', images=uploaded_images)

@app.route('/categorize')
@login_required
def categorize_uploaded_images():
    categorized_images = categorize_images(app.config['UPLOAD_FOLDER'])
    return jsonify(categorized_images)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

# Function to compare two images and highlight differences
def image_difference(image1_path, image2_path, output_path):
    try:
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        if img1.size != img2.size:
            raise Exception("Images must have the same dimensions for comparison.")

        diff = ImageChops.difference(img1, img2)
        if diff.getbbox():
            diff.save(output_path)
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error comparing images: {str(e)}")
        return False

# Function to apply image filters
def apply_image_filter(image_path, filter_name, output_path):
    try:
        img = Image.open(image_path)

        if filter_name == "grayscale":
            img = ImageOps.grayscale(img)
        elif filter_name == "sepia":
            img = ImageOps.colorize(img.convert("L"), "#704214", "#C0A080")
        elif filter_name == "blur":
            img = img.filter(ImageFilter.BLUR)

        img.save(output_path)
    except Exception as e:
        logging.error(f"Error applying image filter: {str(e)}")

# Function to export selected images to a PDF document
def export_images_to_pdf(image_paths, pdf_path):
    try:
        images = [Image.open(image_path) for image_path in image_paths]
        images[0].save(
            pdf_path, save_all=True, append_images=images[1:], resolution=100.0, quality=95, optimize=True
        )
    except Exception as e:
        logging.error(f"Error exporting images to PDF: {str(e)}")

# Machine learning-based image categorization (using a pre-trained model)
def categorize_images(images_directory):
    try:
        # Load a pre-trained image classification model (e.g., MobileNetV2)
        model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')

        # Iterate through images in the directory and classify them
        categorized_images = {}
        for image_file in os.listdir(images_directory):
            image_path = os.path.join(images_directory, image_file)
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=1)[0]
            category = decoded_predictions[0][1]

            if category not in categorized_images:
                categorized_images[category] = []
            categorized_images[category].append(image_path)

        return categorized_images
    except Exception as e:
        logging.error(f"Error categorizing images: {str(e)}")
