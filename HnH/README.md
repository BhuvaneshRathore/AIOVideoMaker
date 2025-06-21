# Video Slideshow Generator

A powerful Flask-based web application that transforms your images into professional video slideshows with customizable transitions, audio, and watermarking.

![Video Generator Preview](https://example.com/preview-image.png)

## Features

- **High-Quality Video Generation**: Create stunning 4K (3840x2160), 4K Vertical (2160x3840), or Instagram Reel (1080x1920) format videos
- **Rich Transition Effects**: Choose from multiple transition types, including:
  - Basic: fade, smooth_cross_fade
  - Wipe: wipe_left, wipe_right, wipe_up, wipe_down
  - Zoom: zoom_in, zoom_out, zoom_push
  - Slide: slide_left, slide_right
  - Split: split_screen_slide, sliding_panels, vertical_blinds
  - Special Effects: pixelize, circular_reveal, vignette_fade, whip_pan, film_burn, glitch, rgb_channel_shift, wave_distortion, prismatic_split, film_grain_overlay, colorize, cinematic_letterbox, ken_burns, edge_detection_blend, ripple
  - Random: picks a random transition for each image change
- **Special First & Last Transitions**: Apply different transitions for the first and last images
- **Professional Watermarking**:
  - Add custom watermarks with adjustable opacity and size
  - 9 predefined watermark positions (corners, edges, center)
  - Custom positioning with X/Y coordinate sliders
  - Preview watermark placement before generating video
- **Audio Support**:
  - Background music support (loops to match video length)
  - Voiceover/speech audio integration
  - Volume balance between music and speech
- **Flexible Image Handling**:
  - Support for multiple image formats (JPG, PNG, GIF, WEBP)
  - Special first and last images
  - Automatic aspect ratio preservation with letterboxing/pillarboxing
  - Batch image upload
  - Directory upload

## System Requirements

- Python 3.6+
- OpenCV (cv2)
- Flask
- MoviePy
- NumPy
- SciPy (optional, for advanced transitions)
- Pillow/PIL (optional, for advanced transitions)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/video-slideshow-generator.git
   cd video-slideshow-generator
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   python app.py
   ```

5. **Open in browser**:
   Open your web browser and navigate to `http://localhost:5000`

## Directory Structure

```
├── app.py              # Flask web application
├── generate.py         # Video generation core functionality
├── static/             # Generated videos and static assets
│   ├── final_output.mp4 # Current output video
│   └── style.css       # CSS styling
├── templates/          # HTML templates
│   ├── index.html      # Main interface
│   └── result.html     # Results page
├── myimg/              # Uploaded images
├── myspeech/           # Uploaded speech/voiceover audio
├── mybackground/       # Uploaded background music
├── Tools/              # Default watermarks and tools
│   ├── logo.PNG        # Default watermark
│   ├── logo1.PNG       # Alternative watermark 1
│   ├── logo2.PNG       # Alternative watermark 2
│   ├── watermark.py    # Watermark application utility
│   ├── Images/         # Custom watermark uploads
│   └── RestAPI/        # Future API capability (not currently implemented)
├── backup_css/         # Backup of unused CSS files
├── backup_js/          # Backup of unused JavaScript files
├── backup_py/          # Backup of unused Python files
├── backup_templates/   # Backup of unused HTML templates
├── backup_videos/      # Backup of older output videos
└── requirements.txt    # Python dependencies
```

## Usage Guide

### 1. Upload Images

- Use the **Upload Images** button to select and upload multiple images
- Or use **Upload Directory** to upload a folder of images
- For special transition effects, upload images named `firstphoto.jpg` and `endimage.jpg`

### 2. Select Video Settings

- **Resolution**: Choose between 4K (16:9), 4K Vertical (9:16), or Instagram Reel (9:16) format
- **Transition**: Select from various transition effects
- **First/Last Special Transition**: Optionally set different transitions for the start and end of the video

### 3. Watermark Settings

- **Enable/Disable**: Toggle watermark visibility
- **Select Image**: Choose from default or uploaded watermarks
- **Position**: Select from predefined positions or use custom X/Y positioning
- **Opacity**: Adjust watermark transparency (0-100%)
- **Size**: Set watermark size as a percentage of the video or use auto sizing
- **Preview**: See how the watermark will appear in the final video

### 4. Audio Settings

- **Background Music**: Upload an MP3 file for background music
- **Voiceover/Speech**: Upload an MP3 file for voiceover or speech

### 5. Generate Video

- Click **Generate Video** to create your slideshow
- The process may take several minutes depending on the number of images and transitions
- Once complete, you'll be redirected to the results page to view and download your video

## Watermark Positioning Guide

The application offers nine predefined watermark positions:

- **Top**: Top Left, Top Center, Top Right
- **Middle**: Center Left, Center, Center Right
- **Bottom**: Bottom Left, Bottom Center, Bottom Right

For custom positioning, select the "Custom Position" option and use the X/Y sliders to place the watermark exactly where you want it.

## Transition Types Explained

### Basic Transitions

- **fade**: Standard crossfade between images
- **smooth_cross_fade**: Enhanced fade with smoother blending

### Directional Transitions

- **wipe_left/right/up/down**: Reveals the next image by wiping from a direction
- **slide_left/right**: Slides the next image in from the side

### Zoom Transitions

- **zoom_in**: Next image grows from the center
- **zoom_out**: Current image shrinks into the next one
- **zoom_push**: Zoom effect with a pushing motion

### Split Transitions

- **split_screen_slide**: Image splits into top and bottom halves
- **sliding_panels**: Horizontal strips slide in alternating directions
- **vertical_blinds**: Vertical strips reveal the next image

### Special Effects

- **pixelize**: Pixelates the current image into the next
- **circular_reveal**: Circular wipe transition
- **film_burn**: Film burning effect at the transition edge
- **glitch**: Digital glitch effect during transition
- **colorize**: Color tinting during transition
- **ripple**: Water ripple effect between images

## Watermark Frontend Implementation

The watermark preview functionality includes:

- Real-time updating as settings change
- Proper positioning that matches the actual video output
- Dynamic calculation of margins based on video dimensions
- Handling of different resolution formats
- Accurate preview of size and transparency


## Acknowledgments

- OpenCV for image processing
- Flask for web framework
- MoviePy for video editing capabilities
- All the open-source contributors who made this possible
