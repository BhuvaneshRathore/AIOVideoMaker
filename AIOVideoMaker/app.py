from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, session, jsonify
import os
from generate import generatex
import datetime
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'slideshowgeneratorapplication'  # Secret key for session
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.config['UPLOAD_FOLDER_img'] = 'myimg'
app.config['UPLOAD_FOLDER_speech'] = 'myspeech'
app.config['UPLOAD_FOLDER_background'] = 'mybackground'
app.config['UPLOAD_FOLDER_watermark'] = 'Tools/Images'  # Directory for watermark images

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER_img'], app.config['UPLOAD_FOLDER_speech'], 
               app.config['UPLOAD_FOLDER_background'], app.config['UPLOAD_FOLDER_watermark']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    # Get list of images
    images = os.listdir(app.config['UPLOAD_FOLDER_img'])
    
    # If we have a stored order in session, use it to sort the images
    stored_order = session.get('image_order', [])
    if stored_order:
        # Filter out any images that no longer exist
        valid_stored_images = [img for img in stored_order if img in images]
        
        # Identify new images that aren't in the stored order
        new_images = [img for img in images if img not in valid_stored_images]
        
        # Combine the ordered list with any new images
        images = valid_stored_images + new_images
    
    has_files = images and os.listdir(app.config['UPLOAD_FOLDER_background'])
    
    # Get watermark images for selection
    watermark_files = []
    if os.path.exists("Tools"):
        # Add default logo from Tools directory
        if os.path.exists(os.path.join("Tools", "logo.PNG")):
            watermark_files.append(("Tools/logo.PNG", "Default Logo"))
        if os.path.exists(os.path.join("Tools", "logo1.PNG")):
            watermark_files.append(("Tools/logo1.PNG", "Logo 1"))
        if os.path.exists(os.path.join("Tools", "logo2.PNG")):
            watermark_files.append(("Tools/logo2.PNG", "Logo 2"))
    
    # Add custom uploaded watermarks
    custom_watermarks = []
    if os.path.exists(app.config['UPLOAD_FOLDER_watermark']):
        for file in os.listdir(app.config['UPLOAD_FOLDER_watermark']):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                watermark_path = os.path.join(app.config['UPLOAD_FOLDER_watermark'], file)
                custom_watermarks.append((watermark_path, f"Custom: {file}"))
    
    # Combine both lists
    watermark_files.extend(custom_watermarks)
    
    return render_template('index.html', images=images, has_files=has_files, watermark_files=watermark_files)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('file')
    
    for file in files:
        if file.filename == '':
            continue

        # Generate a unique filename based on the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER_img'], filename))
    
    return redirect(url_for('index'))

@app.route('/upload_directory', methods=['POST'])
def upload_directory():
    if 'directory' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('directory')
    
    for file in files:
        if file.filename == '' or not file.filename:
            continue
            
        # Skip hidden files and non-image files
        if file.filename.startswith('.') or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            continue
            
        # Generate a unique filename based on the current timestamp to avoid overwriting
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{secure_filename(os.path.basename(file.filename))}"
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER_img'], filename))
    
    return redirect(url_for('index'))

@app.route('/upload_special', methods=['POST'])
def upload_special():
    # Handle first image
    if 'firstphoto' in request.files and request.files['firstphoto'].filename != '':
        first_file = request.files['firstphoto']
        # Remove any existing firstphoto.jpg
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER_img'], 'firstphoto.jpg')):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER_img'], 'firstphoto.jpg'))
        first_file.save(os.path.join(app.config['UPLOAD_FOLDER_img'], 'firstphoto.jpg'))
    
    # Handle last image
    if 'endimage' in request.files and request.files['endimage'].filename != '':
        end_file = request.files['endimage']
        # Remove any existing endimage.jpg
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER_img'], 'endimage.jpg')):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER_img'], 'endimage.jpg'))
        end_file.save(os.path.join(app.config['UPLOAD_FOLDER_img'], 'endimage.jpg'))
    
    return redirect(url_for('index'))

@app.route('/myimg/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_img'], filename)

@app.route('/Tools/<path:filename>')
def serve_tool_file(filename):
    return send_from_directory('Tools', filename)
    
@app.route('/Tools/Images/<path:filename>')
def serve_watermark_file(filename):
    return send_from_directory(os.path.join('Tools', 'Images'), filename)

@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join('static', filename)
    
    # Check if file exists
    if not os.path.exists(video_path):
        return "Video not found", 404
        
    # Send file with proper headers for video streaming
    response = send_from_directory('static', filename)
    response.headers['Content-Type'] = 'video/mp4'
    return response

@app.route('/delete/<image>')
def delete(image):
    os.remove(os.path.join(app.config['UPLOAD_FOLDER_img'], image))
    return redirect(url_for('index'))

@app.route('/reorder_images', methods=['POST'])
def reorder_images():
    """Handle the reordering of images."""
    if request.method == 'POST':
        # Get the new order of images from the POST data
        data = request.get_json()
        new_order = data.get('imageOrder', [])
        
        # For debugging
        print(f"Received new image order: {new_order}")
        
        # Verify all images in the new order exist
        img_folder = app.config['UPLOAD_FOLDER_img']
        existing_images = os.listdir(img_folder)
        valid_order = [img for img in new_order if img in existing_images]
        
        # Store the order in session
        session['image_order'] = valid_order
        
        return jsonify({"status": "success"}), 200
    
    return jsonify({"status": "error", "message": "Invalid request"}), 400

@app.route('/result')
def result():
    """Display the result page with the generated video."""
    output_file = "final_output.mp4"
    output_path = os.path.join("static", output_file)
    
    # Log the access to the result page
    print(f"🎬 Displaying result page with video: {output_path}")
    
    # Check if the file exists and log file info
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📊 Video file exists: {file_size_mb:.2f} MB")
    else:
        print(f"⚠️ Warning: Video file not found at {output_path}")
    
    return render_template(
        'result.html',
        output_path=output_path,
        transition_type="fade",
        first_transition=None,
        last_transition=None
    )

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        background_file = request.files.get('background')
        
        # Clear speech directory before adding new file
        for file in os.listdir(app.config['UPLOAD_FOLDER_speech']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER_speech'], file))
          # Handle speech file - now optional
        speech_file = request.files.get('speech')
        if speech_file and speech_file.filename.endswith('.mp3') and speech_file.filename != '':
            speech_file.save(f"{app.config['UPLOAD_FOLDER_speech']}/{speech_file.filename}")
        
        # Get form parameters
        resolution_type = request.form.get('resolution_type', '4K')
        transition_type = request.form.get('transition_type', 'fade')
        first_transition = request.form.get('first_transition', '')
        last_transition = request.form.get('last_transition', '')
          # Get watermark parameters
        enable_watermark = request.form.get('enable_watermark') == 'yes'
        watermark_path = request.form.get('watermark_path', 'Tools/logo.PNG')
        watermark_position = request.form.get('watermark_position', 'bottom-right')
        watermark_opacity = float(request.form.get('watermark_opacity', '0.7'))
        watermark_size = request.form.get('watermark_size', 'auto')
        
        # Get custom position values if using custom position
        custom_pos_x = None
        custom_pos_y = None
        if watermark_position == 'custom':
            custom_pos_x = request.form.get('custom_pos_x_value')
            custom_pos_y = request.form.get('custom_pos_y_value')
        
        # Handle background file - now optional
        if background_file and background_file.filename.endswith('.mp3'):
            # Clear background directory before adding new file
            for file in os.listdir(app.config['UPLOAD_FOLDER_background']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER_background'], file))
                
            background_file.save(f"{app.config['UPLOAD_FOLDER_background']}/{background_file.filename}")
        else:
            # Make sure the background directory is empty to indicate no background music
            for file in os.listdir(app.config['UPLOAD_FOLDER_background']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER_background'], file))        # Get ordered images from session if available
        ordered_images = session.get('image_order', None)
        
        # Generate the video with selected parameters
        print(f"============================================")
        print(f"Starting video generation with: Resolution={resolution_type}, Transition={transition_type}")
        print(f"First Transition={first_transition}, Last Transition={last_transition}")
        if enable_watermark:
            print(f"Watermark enabled: {watermark_path}, Position: {watermark_position}, Opacity: {watermark_opacity}")
        else:
            print("Watermark disabled")
        print(f"============================================")
        output_path = generatex(
            resolution_type=resolution_type, 
            transition_type=transition_type,
            first_transition=first_transition,
            last_transition=last_transition,
            enable_watermark=enable_watermark,
            watermark_path=watermark_path,
            watermark_position=watermark_position,
            watermark_opacity=watermark_opacity,
            watermark_size=watermark_size,
            custom_pos_x=custom_pos_x,
            custom_pos_y=custom_pos_y,
            ordered_images=ordered_images
        )
        # Make sure output_path exists
        if not os.path.exists(output_path):
            print(f"Warning: Output file {output_path} does not exist!")
        else:
            # Get file size in MB for logging
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"============================================")
            print(f"Generated video successfully at {output_path}")
            print(f"File size: {file_size_mb:.2f} MB")
            print(f"Redirecting to result page...")
            print(f"============================================")
        return redirect(url_for('result'))

    return redirect(url_for('index'))

@app.route('/upload_watermark', methods=['POST'])
def upload_watermark():
    if 'watermark_file' not in request.files:
        return "No file part", 400
    
    file = request.files['watermark_file']
    
    if file.filename == '':
        return "No file selected", 400
        
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER_watermark'], filename))
        return "Success", 200
    else:
        return "Invalid file type", 400
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Force stdout to be line-buffered to ensure console logs are displayed immediately
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    
    # Run Flask with debugging enabled
    app.run(debug=True, port=5000, use_reloader=True)