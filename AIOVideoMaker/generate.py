import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import glob
import random
import math
try:
    from scipy import ndimage
except ImportError:
    print("Warning: scipy not installed. Some advanced transitions may not work.")
try:
    from PIL import Image, ImageFilter, ImageOps, ImageEnhance
except ImportError:
    print("Warning: PIL/Pillow not installed. Some advanced transitions may not work.")
import math
from scipy import ndimage
import colorsys

def generatex(resolution_type="4K", transition_type="fade", first_transition=None, last_transition=None,
           enable_watermark=False, watermark_path="Tools/logo.PNG", watermark_position="bottom-right", 
           watermark_opacity=0.7, watermark_size="auto", custom_pos_x=None, custom_pos_y=None, ordered_images=None):
    """Generate a video slideshow from images with transitions, audio, and optional watermark.
    
    Args:
        resolution_type (str): "4K" for 4K 16:9 resolution, "4K_Vertical" for 4K 9:16 resolution, or "Reel" for Instagram Reel 9:16
        transition_type (str): Type of transition effect between images
        first_transition (str, optional): Special transition for the first image
        last_transition (str, optional): Special transition for the last image
        enable_watermark (bool): Whether to add watermark to the video
        watermark_path (str): Path to the watermark image
        watermark_position (str): Position of watermark ('bottom-right', 'top-left', 'center', etc.)
        ordered_images (list, optional): List of image filenames in the order they should appear in the slideshow
        watermark_opacity (float): Opacity of the watermark (0.0 to 1.0)
        watermark_size (str): Size of watermark ('auto', 'tiny', 'small', 'medium', 'large', 'xlarge', or float between 0.01 and 0.5)
    """
    # Use default transition if special ones not specified
    first_transition = first_transition or transition_type
    last_transition = last_transition or transition_type
    # Directories
    image_directory = 'myimg'
    directory_back = "mybackground"
    directory_speech = "myspeech"
      # Ensure required directories exist
    for directory in [image_directory, directory_back, directory_speech]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")    # Set resolution based on type
    if resolution_type == "Reel" or resolution_type == "Shorts":
        # Optimal resolution for YouTube Shorts / Instagram Reels (9:16 vertical format)
        video_width = 1080   # Standard vertical short-form video width
        video_height = 1920  # Standard vertical short-form video height
        print(f"Creating vertical short-form video ({video_width}x{video_height}) - Optimized for YouTube Shorts/Reels")
    elif resolution_type == "4K_Vertical":
        # 4K Vertical Resolution (9:16 format at 4K quality)
        video_width = 2160   # 4K vertical width for 9:16 aspect ratio
        video_height = 3840  # 4K vertical height for 9:16 aspect ratio
        print(f"Creating 4K Vertical format video ({video_width}x{video_height})")
    elif resolution_type == "1080p":
        # Full HD Resolution (16:9 format)
        video_width = 1920   # 1080p horizontal resolution
        video_height = 1080  # 1080p vertical resolution
        print(f"Creating 1080p format video ({video_width}x{video_height})")
    else:
        # Default 4K Resolution settings
        video_width = 3840   # 4K horizontal resolution
        video_height = 2160  # 4K vertical resolution
        print(f"Creating 4K format video ({video_width}x{video_height}) - Optimal for YouTube")
    
    # Video settings
    frame_rate = 60       # Higher frame rate for smoother transitions
    image_display_duration = 1000  # milliseconds (1 second)
    transition_duration = 60       # Increased for smoother 4K transitions
      # Higher quality video encoding with resolution type in filename
    temp_video_path = f"temp_{resolution_type}_video.mp4"
    final_output_path = "static/final_output.mp4"
    
    # Find audio files
    mp3_files_back = glob.glob(os.path.join(directory_back, "*.mp3"))
    background_audio_path = mp3_files_back[0] if mp3_files_back else None
    print(f"Background audio: {background_audio_path}")
    
    mp3_files_speech = glob.glob(os.path.join(directory_speech, "*.mp3"))
    speech_audio_path = mp3_files_speech[0] if mp3_files_speech else None
    print(f"Speech audio: {speech_audio_path}")
    
    # Set up video writer with H.264 codec for better quality
    fourcc = cv2.VideoWriter_fourcc(*'H264') if cv2.VideoWriter_fourcc(*'H264') else cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(
        temp_video_path, 
        fourcc, 
        frame_rate, 
        (video_width, video_height)
    )
      # Get image files
    image_files = []
    
    if ordered_images:
        # Use the ordered images provided by the user
        print("Using custom image order from user arrangement")
        
        # Validate that all ordered images exist
        valid_ordered_images = [img for img in ordered_images if os.path.exists(os.path.join(image_directory, img))]
        
        # Check if there are any images in the directory that aren't in the ordered list
        all_valid_images = [
            f for f in os.listdir(image_directory) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
        ]
        missing_images = [img for img in all_valid_images if img not in valid_ordered_images]
        
        # Use the ordered images first, then add any missing images
        image_files = valid_ordered_images + missing_images
    else:
        # Use default ordering with special handling for first and last images
        
        # Check for special first image
        first_image_path = os.path.join(image_directory, 'firstphoto.jpg')
        if os.path.exists(first_image_path):
            image_files.append('firstphoto.jpg')
            print("Using firstphoto.jpg as the first image")
        
        # Add all other images except the special ones
        regular_images = [
            f for f in os.listdir(image_directory) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
            and f != 'firstphoto.jpg' and f != 'endimage.jpg'
        ]
        
        # Sort the regular images
        regular_images.sort()
        image_files.extend(regular_images)
        
        # Check for special last image
        last_image_path = os.path.join(image_directory, 'endimage.jpg')
        if os.path.exists(last_image_path):
            image_files.append('endimage.jpg')
            print("Using endimage.jpg as the last image")
    
    if not image_files:
        print("No images found in the directory. Please add images and try again.")
        return
    print(f"Processing {len(image_files)} images at {resolution_type} resolution ({video_width}x{video_height})")
    print(f"Using main transition effect: {transition_type}")
    if first_transition != transition_type:
        print(f"Using special first transition: {first_transition}")
    if last_transition != transition_type:
        print(f"Using special last transition: {last_transition}")
    # Generate video from images with selected transition effect
    for i in range(len(image_files)):
        print(f"Processing image {i+1}/{len(image_files)}: {image_files[i]}")
        
        # Load and prepare current image
        image_path = os.path.join(image_directory, image_files[i])
        current_image = cv2.imread(image_path)
        
        if current_image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
            
        # Resize image to proper resolution with aspect ratio preservation
        current_image = resize_with_aspect_ratio(current_image, width=video_width, height=video_height)
          # Create black background for letterboxing/pillarboxing if needed
        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        
        # Center the image on the frame
        y_offset = (video_height - current_image.shape[0]) // 2
        x_offset = (video_width - current_image.shape[1]) // 2
        frame[
            y_offset:y_offset+current_image.shape[0], 
            x_offset:x_offset+current_image.shape[1]
        ] = current_image
        
        # Apply watermark if enabled
        if enable_watermark and os.path.exists(watermark_path):            frame = apply_watermark_to_frame(
                frame, 
                watermark_path, 
                position=watermark_position,
                opacity=watermark_opacity,
                scale=watermark_size,
                custom_pos_x=custom_pos_x,
                custom_pos_y=custom_pos_y
            )
          # Hold the current image for the specified duration
        for _ in range(int(image_display_duration * frame_rate / 1000)):
            output_video.write(frame)
        
        # Transition to next image if not the last one
        if i < len(image_files) - 1:
            next_image_path = os.path.join(image_directory, image_files[i+1])
            next_image = cv2.imread(next_image_path)
            if next_image is None:
                print(f"Warning: Could not read next image {next_image_path}")
                continue
            
            next_image = resize_with_aspect_ratio(next_image, width=video_width, height=video_height)
            next_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            next_y_offset = (video_height - next_image.shape[0]) // 2
            next_x_offset = (video_width - next_image.shape[1]) // 2
            next_frame[
                next_y_offset:next_y_offset+next_image.shape[0], 
                next_x_offset:next_x_offset+next_image.shape[1]
            ] = next_image
            
            # Apply the appropriate transition effect based on image position
            if i == 0 and image_files[i] == 'firstphoto.jpg' and first_transition:
                # First image with special transition
                print(f"  Using special first transition: {first_transition}")
                apply_transition(frame, next_frame, first_transition, transition_duration, output_video)
            elif i == len(image_files) - 2 and image_files[i+1] == 'endimage.jpg' and last_transition:
                # Last transition to endimage.jpg
                print(f"  Using special last transition: {last_transition}")
                apply_transition(frame, next_frame, last_transition, transition_duration, output_video)
            else:
                # Standard transition for all other images
                apply_transition(frame, next_frame, transition_type, transition_duration, output_video)
    
    output_video.release()
    
    # Get file size for logging
    if os.path.exists(temp_video_path):
        temp_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
        print(f"Video without audio created: {temp_video_path} ({temp_size_mb:.2f} MB)")
    else:
        print(f"WARNING: Expected temp video not found: {temp_video_path}")
    
    print(f"▶▶▶ Adding audio to video... ▶▶▶")
    
    # Add audio to the video
    add_audio_to_video(
        temp_video_path, 
        final_output_path, 
        background_audio_path, 
        speech_audio_path
    )
    
    # Clean up temporary files
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        print(f"✓ Temporary file cleaned up")
    
    # Get final file size for logging
    if os.path.exists(final_output_path):
        final_size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
        print(f"✅ Final {resolution_type} video created: {final_output_path} ({final_size_mb:.2f} MB)")
    else:
        print(f"❌ ERROR: Final video not found at expected path: {final_output_path}")
    return final_output_path

def apply_transition(frame1, frame2, transition_type, duration, output_video):
    """Apply different transition effects between two frames
    
    Args:
        frame1: First frame (current image)
        frame2: Second frame (next image)
        transition_type: Type of transition effect
        duration: Duration of transition in frames
        output_video: VideoWriter object
    """
    h, w = frame1.shape[:2]
    
    # Check if we have Pillow and scipy available for advanced transitions
    has_pillow = 'PIL' in globals() or 'Image' in globals()
    has_scipy = 'ndimage' in globals()
    
    # BASIC TRANSITIONS
    if transition_type == "fade" or transition_type == "smooth_cross_fade":
        # Standard fade transition (cross dissolve)
        for t in range(duration + 1):
            alpha = t / duration
            blended_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            output_video.write(blended_frame)
    
    # DIRECTIONAL WIPES
    elif transition_type == "wipe_left":
        # Wipe from right to left
        for t in range(duration + 1):
            progress = int(w * t / duration)
            result = frame1.copy()
            result[:, w-progress:w, :] = frame2[:, w-progress:w, :]
            output_video.write(result)
    
    elif transition_type == "wipe_right":
        # Wipe from left to right
        for t in range(duration + 1):
            progress = int(w * t / duration)
            result = frame1.copy()
            result[:, 0:progress, :] = frame2[:, 0:progress, :]
            output_video.write(result)
    
    elif transition_type == "wipe_up":
        # Wipe from bottom to top
        for t in range(duration + 1):
            progress = int(h * t / duration)
            result = frame1.copy()
            result[h-progress:h, :, :] = frame2[h-progress:h, :, :]
            output_video.write(result)
    
    elif transition_type == "wipe_down":
        # Wipe from top to bottom
        for t in range(duration + 1):
            progress = int(h * t / duration)
            result = frame1.copy()
            result[0:progress, :, :] = frame2[0:progress, :, :]
            output_video.write(result)
    
    # ZOOM TRANSITIONS
    elif transition_type == "zoom_in":
        # Zoom in transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Create zoomed and centered frame2
            scale = 0.3 + 0.7 * alpha  # Start at 30% size
            zoomed = cv2.resize(frame2, None, fx=scale, fy=scale)
            
            # Center the zoomed image
            result = frame1.copy()
            z_h, z_w = zoomed.shape[:2]
            y_offset = (h - z_h) // 2
            x_offset = (w - z_w) // 2
            
            if y_offset >= 0 and x_offset >= 0:  # Ensure it fits
                result[y_offset:y_offset+z_h, x_offset:x_offset+z_w] = cv2.addWeighted(
                    result[y_offset:y_offset+z_h, x_offset:x_offset+z_w], 1-alpha, zoomed, alpha, 0
                )
            output_video.write(result)
    
    elif transition_type == "zoom_out":
        # Zoom out transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Create oversized frame2 and zoom out
            scale = 1.7 - 0.7 * alpha  # Start at 170% size
            large_frame = cv2.resize(frame2, None, fx=scale, fy=scale)
            
            # Get the center portion
            l_h, l_w = large_frame.shape[:2]
            y_offset = (l_h - h) // 2
            x_offset = (l_w - w) // 2
            
            if y_offset >= 0 and x_offset >= 0 and y_offset+h <= l_h and x_offset+w <= l_w:
                zoomed = large_frame[y_offset:y_offset+h, x_offset:x_offset+w]
                result = cv2.addWeighted(frame1, 1-alpha, zoomed, alpha, 0)
                output_video.write(result)
            else:
                # Fallback if dimensions are wrong
                output_video.write(cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0))
    
    elif transition_type == "zoom_push":
        # Zoom Push - First image zooms in while second image pushes in
        for t in range(duration + 1):
            alpha = t / duration
            
            # Create zoomed frame1 (zooming in)
            scale1 = 1.0 + 0.3 * alpha  # Zoom to 130%
            zoomed1 = cv2.resize(frame1, None, fx=scale1, fy=scale1)
            
            # Get the center portion
            z1_h, z1_w = zoomed1.shape[:2]
            y1_offset = (z1_h - h) // 2
            x1_offset = (z1_w - w) // 2
            
            if y1_offset >= 0 and x1_offset >= 0 and y1_offset+h <= z1_h and x1_offset+w <= z1_w:
                zoomed1 = zoomed1[y1_offset:y1_offset+h, x1_offset:x1_offset+w]
            else:
                zoomed1 = frame1  # Fallback
                
            # Create zoomed frame2 (pushing in)
            scale2 = 0.7 + 0.3 * alpha  # Start at 70% and grow to 100%
            zoomed2 = cv2.resize(frame2, None, fx=scale2, fy=scale2)
            
            # Center the second image
            result = np.zeros_like(frame1)
            z2_h, z2_w = zoomed2.shape[:2]
            y2_offset = (h - z2_h) // 2
            x2_offset = (w - z2_w) // 2
            
            if y2_offset >= 0 and x2_offset >= 0:
                result[y2_offset:y2_offset+z2_h, x2_offset:x2_offset+z2_w] = zoomed2
            else:
                result = frame2  # Fallback
                
            # Blend the two effects
            final = cv2.addWeighted(zoomed1, 1-alpha, result, alpha, 0)
            output_video.write(final)
    
    # SLIDING TRANSITIONS
    elif transition_type == "slide_left":
        # Slide left transition
        for t in range(duration + 1):
            progress = int(w * t / duration)
            result = np.zeros_like(frame1)
            
            # Copy part of frame1 (moving left)
            if progress < w:
                result[:, 0:w-progress, :] = frame1[:, progress:w, :]
            
            # Copy part of frame2 (entering from right)
            if progress > 0:
                result[:, w-progress:, :] = frame2[:, 0:progress, :]
                
            output_video.write(result)
    
    elif transition_type == "slide_right":
        # Slide right transition
        for t in range(duration + 1):
            progress = int(w * t / duration)
            result = np.zeros_like(frame1)
            
            # Copy part of frame1 (moving right)
            if progress < w:
                result[:, progress:w, :] = frame1[:, 0:w-progress, :]
            
            # Copy part of frame2 (entering from left)
            if progress > 0:
                result[:, 0:progress, :] = frame2[:, w-progress:w, :]
                
            output_video.write(result)
    
    elif transition_type == "split_screen_slide":
        # Split Screen Slide - Image splits into top and bottom halves
        for t in range(duration + 1):
            progress = int(w * t / duration)
            result = np.zeros_like(frame1)
            
            # Divide height in half
            half_h = h // 2
            
            # Top half moves left
            if progress < w:
                result[:half_h, 0:w-progress, :] = frame1[:half_h, progress:w, :]
            
            # Bottom half moves right
            if progress < w:
                result[half_h:, progress:w, :] = frame1[half_h:, 0:w-progress, :]
            
            # New image coming in from sides
            if progress > 0:
                result[:half_h, w-progress:, :] = frame2[:half_h, 0:progress, :]
                result[half_h:, 0:progress, :] = frame2[half_h:, w-progress:w, :]
            
            output_video.write(result)
    
    elif transition_type == "sliding_panels":
        # Sliding panels transition (horizontal strips)
        num_panels = 5
        panel_height = h // num_panels
        
        for t in range(duration + 1):
            alpha = t / duration
            progress = int(w * alpha)
            result = frame1.copy()
            
            for i in range(num_panels):
                panel_y = i * panel_height
                panel_h = min(panel_height, h - panel_y)
                
                # Alternate direction for each panel
                if i % 2 == 0:  # Even panels slide right
                    if progress > 0:
                        result[panel_y:panel_y+panel_h, 0:progress, :] = frame2[panel_y:panel_y+panel_h, w-progress:w, :]
                else:  # Odd panels slide left
                    if progress > 0:
                        result[panel_y:panel_y+panel_h, w-progress:w, :] = frame2[panel_y:panel_y+panel_h, 0:progress, :]
            
            output_video.write(result)
    
    elif transition_type == "vertical_blinds":
        # Vertical blinds transition
        num_blinds = 10
        blind_width = w // num_blinds
        
        for t in range(duration + 1):
            alpha = t / duration
            progress = int(h * alpha)
            result = frame1.copy()
            
            for i in range(num_blinds):
                blind_x = i * blind_width
                blind_w = min(blind_width, w - blind_x)
                
                # All blinds slide down
                if progress > 0:
                    result[0:progress, blind_x:blind_x+blind_w, :] = frame2[h-progress:h, blind_x:blind_x+blind_w, :]
                
            output_video.write(result)
    
    # VISUAL EFFECT TRANSITIONS
    elif transition_type == "pixelize":
        # Pixelization transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Start with large pixels, gradually refine to the actual image
            pixel_size = max(1, int((1 - alpha) * 30))
            
            if pixel_size > 1:
                # Downscale and upscale frame2 to create pixelization effect
                small = cv2.resize(frame2, (w // pixel_size, h // pixel_size))
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Blend the pixelated image with frame1
                blend_alpha = min(1.0, alpha * 2)  # Double speed for alpha blend
                result = cv2.addWeighted(frame1, 1 - blend_alpha, pixelated, blend_alpha, 0)
            else:
                # At the end, just blend to the final image
                result = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                
            output_video.write(result)
    
    elif transition_type == "gaussian_blur" and has_scipy:
        # Gaussian Blur transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Gradually blur the first image
            blur_sigma = alpha * 10.0  # Max blur sigma
            blurred1 = ndimage.gaussian_filter(frame1, sigma=(blur_sigma, blur_sigma, 0))
            
            # Gradually sharpen the second image from blur
            sharp_sigma = (1.0 - alpha) * 10.0
            blurred2 = ndimage.gaussian_filter(frame2, sigma=(sharp_sigma, sharp_sigma, 0))
            
            # Blend between them
            result = cv2.addWeighted(blurred1, 1 - alpha, blurred2, alpha, 0)
            output_video.write(result)
    
    elif transition_type == "circular_reveal":
        # Circular reveal transition
        center_y, center_x = h // 2, w // 2
        max_radius = int(math.sqrt(w*w + h*h) / 2)  # Maximum radius to cover the entire frame
        
        for t in range(duration + 1):
            alpha = t / duration
            radius = int(alpha * max_radius)
            
            # Create a circular mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # -1 fills the circle
            
            # Apply mask
            result = frame1.copy()
            for c in range(3):  # Apply to each color channel
                result[:,:,c] = np.where(mask == 255, frame2[:,:,c], frame1[:,:,c])
                
            output_video.write(result)
    
    elif transition_type == "vignette_fade":
        # Vignette fade transition
        center_y, center_x = h // 2, w // 2
        max_radius = int(math.sqrt(w*w + h*h) / 2)
        
        for t in range(duration + 1):
            alpha = t / duration
            
            # Create distance matrix from center
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Normalize distances
            max_dist = np.sqrt(center_x**2 + center_y**2)
            norm_dist = dist_from_center / max_dist
            
            # Create vignette mask
            vignette = np.clip(1.0 - norm_dist + alpha, 0, 1)
            
            # Expand mask to 3 channels
            vignette = np.stack([vignette]*3, axis=2)
            
            # Apply vignette mask
            result = frame1.copy() * (1.0 - vignette) + frame2.copy() * vignette
            result = result.astype(np.uint8)
            
            output_video.write(result)
    
    elif transition_type == "whip_pan":
        # Whip Pan - fast motion blur transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Apply horizontal motion blur
            motion_strength = int(30 * math.sin(alpha * math.pi))  # Max blur in middle of transition
            
            # Create a motion-blurred image
            if motion_strength > 0:
                kernel = np.zeros((1, motion_strength))
                kernel[0, :] = 1.0 / motion_strength
                
                # Apply horizontal motion blur to both images
                blurred1 = cv2.filter2D(frame1, -1, kernel)
                blurred2 = cv2.filter2D(frame2, -1, kernel)
                
                # Blend between them with faster movement in the middle
                ease = 0.5 - 0.5 * math.cos(alpha * math.pi * 2)  # Easing function
                result = cv2.addWeighted(blurred1, 1.0 - ease, blurred2, ease, 0)
            else:
                # Normal crossfade at beginning and end
                result = cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)
                
            output_video.write(result)
    
    elif transition_type == "film_burn":
        # Film Burn transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Create a gradient that starts from bottom and moves up
            burn_height = int((1.0 - alpha) * h)
            
            # Copy frames
            result = np.zeros_like(frame1)
            
            # Add first frame below burn line
            if burn_height > 0:
                result[burn_height:, :, :] = frame1[burn_height:, :, :]
            
            # Add second frame above burn line
            if burn_height < h:
                result[:burn_height, :, :] = frame2[:burn_height, :, :]
            
            # Add orange/red burn effect at transition line
            burn_width = 20
            start = max(0, burn_height - burn_width)
            end = min(h, burn_height + burn_width)
            
            if start < end:
                # Create gradient
                for y in range(start, end):
                    intensity = 1.0 - abs(y - burn_height) / burn_width
                    # Add orange/red glow
                    result[y, :, 0] = np.minimum(255, result[y, :, 0] + int(50 * intensity))  # B
                    result[y, :, 1] = np.minimum(255, result[y, :, 1] + int(150 * intensity)) # G
                    result[y, :, 2] = 255  # R (full brightness)
            
            output_video.write(result)
    
    elif transition_type == "glitch":
        # Glitch effect transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Start with blend of two images
            result = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            
            # Apply stronger glitch effects in the middle of the transition
            glitch_intensity = math.sin(alpha * math.pi)
            if glitch_intensity > 0.1:
                num_glitches = int(10 * glitch_intensity)
                
                for _ in range(num_glitches):
                    # Random horizontal slice
                    glitch_y = random.randint(0, h - 20)
                    glitch_h = random.randint(5, 20)
                    glitch_shift = random.randint(-50, 50)
                    
                    # Shift a slice horizontally
                    if glitch_shift > 0:
                        result[glitch_y:glitch_y+glitch_h, glitch_shift:, :] = result[glitch_y:glitch_y+glitch_h, :-glitch_shift, :]
                        # Fill the gap with random color
                        result[glitch_y:glitch_y+glitch_h, :glitch_shift, :] = random.randint(0, 255)
                    elif glitch_shift < 0:
                        result[glitch_y:glitch_y+glitch_h, :w+glitch_shift, :] = result[glitch_y:glitch_y+glitch_h, -glitch_shift:, :]
                        # Fill the gap with random color
                        result[glitch_y:glitch_y+glitch_h, w+glitch_shift:, :] = random.randint(0, 255)
                
                # Add RGB channel shift
                shift_amount = int(5 * glitch_intensity)
                if shift_amount > 0:
                    # Shift red channel
                    result[:, shift_amount:, 2] = result[:, :-shift_amount, 2]
                    # Shift blue channel the opposite way
                    result[:, :-shift_amount, 0] = result[:, shift_amount:, 0]
            
            output_video.write(result)
    
    elif transition_type == "rgb_channel_shift":
        # RGB Channel Shift transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Create result with frame1 fading out
            result1 = frame1.copy() * (1 - alpha)
            
            # Create result with frame2 fading in, with RGB channel shifts
            result2 = np.zeros_like(frame2, dtype=np.float32)
            
            # Calculate channel shifts based on transition progress
            shift_amount = int(20 * math.sin(alpha * math.pi))  # Max 20 pixels, peaking in the middle
            
            # Apply different shifts to each RGB channel
            for c in range(3):
                if c == 0:  # Blue channel
                    shift = shift_amount
                elif c == 1:  # Green channel
                    shift = 0
                else:  # Red channel
                    shift = -shift_amount
                
                # Apply shift
                if shift > 0 and shift < w:
                    result2[:, shift:, c] = frame2[:, :-shift, c] * alpha
                elif shift < 0 and -shift < w:
                    result2[:, :w+shift, c] = frame2[:, -shift:, c] * alpha
                else:
                    result2[:, :, c] = frame2[:, :, c] * alpha
            
            # Combine the results
            final_result = result1 + result2
            final_result = np.clip(final_result, 0, 255).astype(np.uint8)
            
            output_video.write(final_result)
    
    elif transition_type == "wave_distortion":
        # Wave distortion transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Start with blend of two images
            base = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            result = np.zeros_like(base)
            
            # Wave parameters
            wave_amplitude = int(20 * math.sin(alpha * math.pi))  # Max amplitude in middle of transition
            wave_frequency = 10.0  # Number of waves
            wave_phase = alpha * 2 * math.pi  # Phase shifts over transition
            
            # Apply horizontal wave distortion
            for y in range(h):
                # Calculate wave offset for this row
                offset = int(wave_amplitude * math.sin(y / h * wave_frequency * math.pi + wave_phase))
                
                # Apply offset, wrapping around if necessary
                for x in range(w):
                    src_x = (x + offset) % w
                    result[y, x] = base[y, src_x]
            
            output_video.write(result)
    
    elif transition_type == "prismatic_split":
        # Prismatic split transition (RGB separation and recombination)
        for t in range(duration + 1):
            alpha = t / duration
            
            # Split channels of first image
            b1, g1, r1 = cv2.split(frame1)
            
            # Split channels of second image
            b2, g2, r2 = cv2.split(frame2)
            
            # Calculate shift amount for prismatic effect
            shift = int(30 * math.sin(alpha * math.pi))  # Max 30 pixels, peaking in middle
            
            # Create shifted versions of the channels
            shifted_r1 = np.zeros_like(r1)
            shifted_g1 = np.zeros_like(g1)
            shifted_b1 = np.zeros_like(b1)
            
            shifted_r2 = np.zeros_like(r2)
            shifted_g2 = np.zeros_like(g2)
            shifted_b2 = np.zeros_like(b2)
            
            # Apply horizontal shifts
            # Red shifts right
            if shift > 0:
                shifted_r1[:, shift:] = r1[:, :-shift]
                shifted_r2[:, shift:] = r2[:, :-shift]
            else:
                shifted_r1 = r1
                shifted_r2 = r2
                
            # Blue shifts left
            if shift > 0:
                shifted_b1[:, :-shift] = b1[:, shift:]
                shifted_b2[:, :-shift] = b2[:, shift:]
            else:
                shifted_b1 = b1
                shifted_b2 = b2
            
            # Green stays centered
            shifted_g1 = g1
            shifted_g2 = g2
            
            # Recombine channels with alpha blend
            r = cv2.addWeighted(shifted_r1, 1 - alpha, shifted_r2, alpha, 0)
            g = cv2.addWeighted(shifted_g1, 1 - alpha, shifted_g2, alpha, 0)
            b = cv2.addWeighted(shifted_b1, 1 - alpha, shifted_b2, alpha, 0)
            
            result = cv2.merge([b, g, r])
            output_video.write(result)
    
    elif transition_type == "film_grain_overlay":
        # Film grain overlay transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Basic crossfade
            result = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            
            # Add film grain
            grain_intensity = 25 * math.sin(alpha * math.pi)  # Stronger in the middle
            grain = np.random.normal(0, grain_intensity, result.shape).astype(np.int16)
            
            # Add grain to the image
            grainy = cv2.add(result.astype(np.int16), grain, dtype=cv2.CV_8U)
            grainy = np.clip(grainy, 0, 255).astype(np.uint8)
            
            output_video.write(grainy)
    
    elif transition_type == "colorize":
        # Colorize transition - tint the images during transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Crossfade between images
            result = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            
            # Apply color tint that changes through transition
            # Stronger in middle, fades at start and end
            tint_strength = math.sin(alpha * math.pi) * 0.7  # Max 70% tint
            
            # Use hue that shifts during transition (blue to purple to red)
            hue = (240 + int(alpha * 120)) % 360  # 240° (blue) to 0° (red)
            
            # Convert hue to BGR
            h = hue / 60.0
            i = int(h)
            f = h - i
            
            p = 0
            q = 1 - f
            t = f
            
            r, g, b = 0, 0, 0
            
            if i == 0:
                r, g, b = 1, t, p
            elif i == 1:
                r, g, b = q, 1, p
            elif i == 2:
                r, g, b = p, 1, t
            elif i == 3:
                r, g, b = p, q, 1
            elif i == 4:
                r, g, b = t, p, 1
            else:
                r, g, b = 1, p, q
            
            # Apply tint
            tint = np.array([b*255, g*255, r*255], dtype=np.uint8)
            tinted = cv2.addWeighted(result, 1.0 - tint_strength, np.ones_like(result) * tint, tint_strength, 0)
            
            output_video.write(tinted)
    
    elif transition_type == "cinematic_letterbox":
        # Cinematic letterbox transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Create letterboxed images with black bars
            letterbox_height = int((1.0 - math.sin(alpha * math.pi / 2)) * h * 0.15)  # Height of black bars
            
            # Add letterbox to both images
            if letterbox_height > 0:
                # Apply to first image
                letterboxed1 = frame1.copy()
                letterboxed1[:letterbox_height, :] = [0, 0, 0]  # Top bar
                letterboxed1[-letterbox_height:, :] = [0, 0, 0]  # Bottom bar
                
                # Apply to second image
                letterboxed2 = frame2.copy()
                letterboxed2[:letterbox_height, :] = [0, 0, 0]
                letterboxed2[-letterbox_height:, :] = [0, 0, 0]
                
                # Crossfade between letterboxed images
                result = cv2.addWeighted(letterboxed1, 1 - alpha, letterboxed2, alpha, 0)
            else:
                # Just crossfade at the end
                result = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            
            output_video.write(result)
    
    elif transition_type == "ken_burns":
        # Ken Burns effect transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # First image: zoom out and pan
            scale1 = 1.2 - 0.2 * alpha  # Start at 120% and zoom out
            zoomed1 = cv2.resize(frame1, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_CUBIC)
            
            # Pan from top-left to center
            z1_h, z1_w = zoomed1.shape[:2]
            x1_offset = int((z1_w - w) * (1 - alpha))  # Start with offset, move to center
            y1_offset = int((z1_h - h) * (1 - alpha))
            
            # Extract visible portion
            if z1_h > h and z1_w > w and x1_offset >= 0 and y1_offset >= 0:
                cropped1 = zoomed1[y1_offset:y1_offset+h, x1_offset:x1_offset+w]
                if cropped1.shape[:2] == (h, w):
                    frame1_effect = cropped1
                else:
                    frame1_effect = frame1  # Fallback
            else:
                frame1_effect = frame1  # Fallback
            
            # Second image: zoom in and pan
            scale2 = 1.0 + 0.2 * alpha  # Start at 100% and zoom in to 120%
            zoomed2 = cv2.resize(frame2, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_CUBIC)
            
            # Pan from center to bottom-right
            z2_h, z2_w = zoomed2.shape[:2]
            x2_offset = int((z2_w - w) * alpha)  # Start at center, move to max offset
            y2_offset = int((z2_h - h) * alpha)
            
            # Extract visible portion
            if z2_h > h and z2_w > w and x2_offset >= 0 and y2_offset >= 0:
                cropped2 = zoomed2[y2_offset:y2_offset+h, x2_offset:x2_offset+w]
                if cropped2.shape[:2] == (h, w):
                    frame2_effect = cropped2
                else:
                    frame2_effect = frame2  # Fallback
            else:
                frame2_effect = frame2  # Fallback
            
            # Blend between the two effects
            result = cv2.addWeighted(frame1_effect, 1-alpha, frame2_effect, alpha, 0)
            output_video.write(result)
    
    elif transition_type == "edge_detection_blend":
        # Edge detection blend transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Detect edges in both frames
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edge1 = cv2.Canny(gray1, 50, 150)
            edge2 = cv2.Canny(gray2, 50, 150)
            
            # Convert back to BGR
            edge1_bgr = cv2.cvtColor(edge1, cv2.COLOR_GRAY2BGR)
            edge2_bgr = cv2.cvtColor(edge2, cv2.COLOR_GRAY2BGR)
            
            # Determine edge intensity based on transition point
            edge_intensity = math.sin(alpha * math.pi)
            blend_intensity = 1.0 - edge_intensity
            
            # Base crossfade
            result = cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
            
            # Add edges with intensity
            if edge_intensity > 0:
                edge_blend = cv2.addWeighted(edge1_bgr, (1-alpha) * edge_intensity, edge2_bgr, alpha * edge_intensity, 0)
                result = cv2.addWeighted(result, blend_intensity, edge_blend, edge_intensity, 0)
            
            output_video.write(result)
    
    elif transition_type == "ripple":
        # Ripple effect transition
        for t in range(duration + 1):
            alpha = t / duration
            
            # Start with blend of two images
            result = np.zeros_like(frame1)
            
            # Ripple parameters
            center_x, center_y = w // 2, h // 2
            max_dist = math.sqrt(center_x**2 + center_y**2)
            
            # Wave parameters vary with transition progress
            wavelength = 30.0 * (1.0 - alpha * 0.5)  # Decreasing wavelength
            amplitude = 10.0 * math.sin(alpha * math.pi)  # Amplitude peaks in middle
            
            # Generate ripple effect
            for y in range(h):
                for x in range(w):
                    # Distance from center
                    dx = x - center_x
                    dy = y - center_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Create ripple offset
                    offset = amplitude * math.sin(distance / wavelength * 2 * math.pi)
                    
                    # Calculate source coordinates with ripple
                    angle = math.atan2(dy, dx)
                    source_x = int(x + offset * math.cos(angle))
                    source_y = int(y + offset * math.sin(angle))
                    
                    # Keep coordinates in bounds
                    source_x = max(0, min(source_x, w-1))
                    source_y = max(0, min(source_y, h-1))
                    
                    # Crossfade between images with ripple
                    result[y, x] = frame1[source_y, source_x] * (1-alpha) + frame2[source_y, source_x] * alpha
            
            output_video.write(result)
    
    elif transition_type == "random":
        # Choose a random transition from the available options
        transitions = [
            "fade", "smooth_cross_fade", "wipe_left", "wipe_right", "wipe_up", "wipe_down", 
            "zoom_in", "zoom_out", "zoom_push", "slide_left", "slide_right", "split_screen_slide",
            "sliding_panels", "pixelize", "circular_reveal", "vignette_fade", "whip_pan",
            "film_burn", "glitch", "rgb_channel_shift", "wave_distortion", "prismatic_split",
            "film_grain_overlay", "colorize", "cinematic_letterbox", "ken_burns",
            "edge_detection_blend", "ripple", "vertical_blinds"
        ]
        
        # Filter out transitions that need scipy if not available
        if not has_scipy:
            transitions = [t for t in transitions if t != "gaussian_blur"]
        
        random_transition = random.choice(transitions)
        print(f"  Using random transition: {random_transition}")
        apply_transition(frame1, frame2, random_transition, duration, output_video)
    
    else:
        # Fallback to standard fade transition
        for t in range(duration + 1):
            alpha = t / duration
            blended_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            output_video.write(blended_frame)

def resize_with_aspect_ratio(image, width=None, height=None):
    """Resize image preserving aspect ratio with letterboxing/pillarboxing as needed."""
    h, w = image.shape[:2]
    
    if width and height:
        # Calculate the scaling factor to fit within width and height
        scale_w = width / w
        scale_h = height / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the image using high-quality interpolation
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create a black canvas with target dimensions
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        x_offset = (width - new_w) // 2
        y_offset = (height - new_h) // 2
        
        # Place the resized image on the canvas (letterboxing/pillarboxing)
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        
        return canvas
    
    return image

def add_audio_to_video(video_path, output_path, background_audio_path, speech_audio_path):
    """Add background music and speech to the video."""
    print(f"📝 Audio addition details:")
    print(f"  → Video source: {video_path}")
    print(f"  → Output destination: {output_path}")
    print(f"  → Background audio: {'Yes' if background_audio_path else 'No'}")
    print(f"  → Speech audio: {'Yes' if speech_audio_path else 'No'}")
    
    if not (background_audio_path or speech_audio_path):
        print("🔹 No audio files found. Skipping audio addition.")
        # Check if output file already exists and remove it before renaming
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"🗑️ Removed existing file: {output_path}")
        
        # Rename the temporary video file to final output
        print(f"🔄 Renaming {video_path} to {output_path}")
        os.rename(video_path, output_path)
        print(f"✓ Video renamed successfully")
        return
    
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration
    
    audio_clips = []
    
    # Process background audio if available
    if background_audio_path:
        background_audio = AudioFileClip(background_audio_path)
        
        if video_duration > background_audio.duration:
            # Loop background audio to match video length
            repeats_needed = int(np.ceil(video_duration / background_audio.duration))
            background_segments = [background_audio] * repeats_needed
            background_audio = CompositeAudioClip(background_segments)
            background_audio = background_audio.subclip(0, video_duration)
        else:
            background_audio = background_audio.subclip(0, video_duration)
        
        # Add background at reduced volume
        audio_clips.append(background_audio.volumex(0.3))
      # Process speech audio if available (now optional)
    if speech_audio_path:
        speech_audio = AudioFileClip(speech_audio_path)
        
        if video_duration < speech_audio.duration:
            # Trim speech if longer than video
            speech_audio = speech_audio.subclip(0, video_duration)
        
        # Add speech at full volume
        audio_clips.append(speech_audio.volumex(1.0))
    
    if audio_clips:
        final_audio = CompositeAudioClip(audio_clips)
        final_clip = video_clip.set_audio(final_audio)
    else:
        final_clip = video_clip
    
    # Check if output file already exists and remove it before writing
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")
    
    # Write the final video with high-quality settings
    # Adjust bitrate based on resolution and format
    if "4K" in video_path:  # Check if it's a 4K video
        video_bitrate = '40000k'  # Increased bitrate for 4K long videos (YouTube recommendation)
    elif "Reel" in video_path or "Shorts" in video_path:
        video_bitrate = '15000k'  # Optimized for YouTube Shorts/Reels
    else:
        video_bitrate = '24000k'  # Standard high quality
    
    # Set audio bitrate based on format
    if "Reel" in video_path or "Shorts" in video_path:
        audio_bitrate = '192k'  # Ideal for short-form content
    else:
        audio_bitrate = '320k'  # Higher quality for long videos
    
    final_clip.write_videofile(
        output_path, 
        codec='libx264',
        audio_codec='aac',
        bitrate=video_bitrate,
        audio_bitrate=audio_bitrate,
        fps=video_clip.fps,
        threads=4,
        preset='slow',
        ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']  # Optimal for YouTube upload
    )
    # Close clips to release resources
    video_clip.close()

def resize_logo_sharp(logo, target_size):
    """Resize logo while maintaining maximum sharpness"""
    original_width, original_height = logo.size
    target_width, target_height = target_size
    
    # Calculate scaling factors
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    avg_scale = (width_scale + height_scale) / 2
    
    # Choose optimal resampling method
    if avg_scale > 1.0:
        # Upscaling
        resampling = Image.Resampling.LANCZOS
    elif avg_scale > 0.5:
        # Moderate downscaling
        resampling = Image.Resampling.LANCZOS
    else:
        # Heavy downscaling
        resampling = Image.Resampling.LANCZOS
    
    # For very small logos, ensure minimum quality
    if target_width < 50 or target_height < 50:
        resampling = Image.Resampling.BICUBIC
    
    return logo.resize(target_size, resampling)

def apply_opacity_sharp(logo, opacity):
    """Apply opacity while preserving logo sharpness"""
    if logo.mode != 'RGBA':
        logo = logo.convert('RGBA')
    
    if opacity >= 1.0:
        return logo
    
    # Split channels
    r, g, b, a = logo.split()
    
    # Apply opacity to alpha channel
    alpha_factor = int(opacity * 255) / 255
    a = a.point(lambda x: int(x * alpha_factor))
    
    # Merge channels back
    return Image.merge('RGBA', (r, g, b, a))

def apply_watermark_to_frame(frame, watermark_path, position='bottom-right', 
                           opacity=0.7, scale='auto', custom_pos_x=None, custom_pos_y=None):
    """Apply watermark to a video frame
    
    Args:
        frame: numpy array frame from cv2
        watermark_path: path to watermark PNG image
        position: position of watermark ('bottom-right', 'top-left', etc.)
        opacity: opacity of watermark (0.0 to 1.0)
        scale: scale of watermark: 
              - 'auto': Automatically sized based on resolution
              - 'tiny': Very small (5% of frame width)
              - 'small': Small (7% of frame width)
              - 'medium': Medium (10% of frame width)
              - 'large': Large (15% of frame width)
              - 'xlarge': Extra large (20% of frame width)
              - float between 0.01 and 0.5: Custom scale (percent of frame width)
        
    Returns:
        numpy array: frame with watermark
    """
    # Convert cv2 frame (numpy array) to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    try:
        # Open the logo
        logo = Image.open(watermark_path)
        
        # Ensure logo is RGBA for transparency
        if logo.mode != 'RGBA':
            logo = logo.convert('RGBA')
          # Get frame dimensions
        frame_width, frame_height = frame_pil.size
        
        # Determine watermark scale
        if scale == 'auto':
            # Auto-scale based on frame resolution
            if frame_width >= 3840:  # 4K
                watermark_scale = 0.08
            elif frame_width >= 1920:  # Full HD
                watermark_scale = 0.10
            else:
                watermark_scale = 0.12
        elif scale == 'tiny':
            watermark_scale = 0.05
        elif scale == 'small':
            watermark_scale = 0.07
        elif scale == 'medium':
            watermark_scale = 0.10
        elif scale == 'large':
            watermark_scale = 0.15
        elif scale == 'xlarge':
            watermark_scale = 0.20
        else:
            # Try to convert to float, default to auto if it fails
            try:
                watermark_scale = float(scale)
                # Clamp between reasonable values (0.01 to 0.5)
                watermark_scale = max(0.01, min(0.5, watermark_scale))
            except ValueError:
                # If conversion fails, use auto scale based on resolution
                if frame_width >= 3840:  # 4K
                    watermark_scale = 0.08
                elif frame_width >= 1920:  # Full HD
                    watermark_scale = 0.10
                else:
                    watermark_scale = 0.12
          # Calculate watermark size
        watermark_width = int(frame_width * watermark_scale)
        watermark_height = int(logo.height * (watermark_width / logo.width))
        
        # Ensure reasonable size limits
        max_watermark_size = min(frame_width // 2, frame_height // 2)  # Increased max limit
        if watermark_width > max_watermark_size:
            watermark_width = max_watermark_size
            watermark_height = int(logo.height * (watermark_width / logo.width))
        
        # Minimum size for visibility based on resolution
        if frame_width >= 3840:  # 4K
            min_watermark_size = 48  # Larger min size for high-res
        elif frame_width >= 1920:  # Full HD
            min_watermark_size = 40  # Medium min size
        else:
            min_watermark_size = 24  # Smaller min size for lower res
            
        if watermark_width < min_watermark_size:
            watermark_width = min_watermark_size
            watermark_height = int(logo.height * (watermark_width / logo.width))
        
        # Resize logo
        logo = resize_logo_sharp(logo, (watermark_width, watermark_height))
        
        # Apply opacity
        logo = apply_opacity_sharp(logo, opacity)
        
        # Calculate margin based on resolution
        if frame_width >= 3840:  # 4K
            margin_px = int(frame_width * 0.015)
        elif frame_width >= 1920:  # Full HD
            margin_px = int(frame_width * 0.02)
        else:        margin_px = max(20, int(frame_width * 0.025))
        
        # Calculate position
        positions = {
            'top-left': (margin_px, margin_px),
            'top-right': (frame_width - watermark_width - margin_px, margin_px),
            'bottom-left': (margin_px, frame_height - watermark_height - margin_px),
            'bottom-right': (frame_width - watermark_width - margin_px, 
                          frame_height - watermark_height - margin_px),
            'center': ((frame_width - watermark_width) // 2, 
                      (frame_height - watermark_height) // 2),
            'bottom-center': ((frame_width - watermark_width) // 2,
                            frame_height - watermark_height - margin_px)
        }
        
        # Check if custom position is being used
        if position == 'custom' and custom_pos_x is not None and custom_pos_y is not None:
            try:                # Convert percentage to pixels
                pos_x_percent = float(custom_pos_x) / 100.0
                pos_y_percent = float(custom_pos_y) / 100.0
                
                # Calculate pixel position, accounting for watermark dimensions
                pos_x = int((frame_width - watermark_width) * pos_x_percent)
                pos_y = int((frame_height - watermark_height) * pos_y_percent)
                
                # Create a custom position tuple
                watermark_position = (pos_x, pos_y)
            except (ValueError, TypeError):
                # Fall back to default if there's an error
                watermark_position = positions['bottom-right']
        else:
            # Use predefined positions
            watermark_position = positions.get(position, positions['bottom-right'])
        
        # Create a transparent overlay
        overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
        
        # Paste logo with alpha channel
        overlay.paste(logo, watermark_position, logo)
        
        # Composite images
        watermarked = Image.alpha_composite(frame_pil.convert('RGBA'), overlay)
        
        # Convert back to BGR for cv2
        result_frame = cv2.cvtColor(np.array(watermarked.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        return result_frame
        
    except Exception as e:
        print(f"Error applying watermark: {e}")
        # Return original frame if watermarking fails
        return frame

if __name__ == "__main__":
    generatex()