#!/usr/bin/env python3
"""
Image Watermark Script
Adds a PNG logo as watermark to all images in a directory
OPTIMIZED FOR SHARP, HIGH-QUALITY LOGO RENDERING
"""

import os
import sys
from PIL import Image, ImageEnhance, ImageOps, ImageStat, ImageFilter
import argparse
from pathlib import Path
import colorsys
import numpy as np

def analyze_background_color(image, position, watermark_size):
    """
    Analyze the background color where the watermark will be placed
    
    Args:
        image: PIL Image object
        position: (x, y) position where watermark will be placed
        watermark_size: (width, height) of the watermark
        
    Returns:
        dict: Background analysis with average color, brightness, and contrast recommendations
    """
    x, y = position
    w, h = watermark_size
    
    # Extract the region where watermark will be placed
    # Add some padding to get a better sample
    padding = 10
    sample_region = image.crop((
        max(0, x - padding),
        max(0, y - padding),
        min(image.width, x + w + padding),
        min(image.height, y + h + padding)
    ))
    
    # Convert to RGB for analysis
    if sample_region.mode != 'RGB':
        sample_region = sample_region.convert('RGB')
    
    # Get average color
    stat = ImageStat.Stat(sample_region)
    avg_color = tuple(int(c) for c in stat.mean)
    
    # Calculate brightness (0-255)
    brightness = sum(avg_color) / 3
    
    # Calculate perceived brightness using luminance formula
    r, g, b = avg_color
    perceived_brightness = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Determine if background is light or dark
    is_light_background = perceived_brightness > 127
    
    # Calculate color variance to determine if background is uniform
    variance = sum((c - brightness) ** 2 for c in avg_color) / 3
    is_uniform = variance < 1000  # Threshold for uniform background
    
    return {
        'avg_color': avg_color,
        'brightness': brightness,
        'perceived_brightness': perceived_brightness,
        'is_light': is_light_background,
        'is_uniform': is_uniform,
        'variance': variance
    }

def adjust_logo_contrast(logo, background_analysis, contrast_mode='auto'):
    """
    Adjust logo colors for optimal contrast against background
    OPTIMIZED TO PRESERVE LOGO SHARPNESS
    
    Args:
        logo: PIL Image object (logo)
        background_analysis: dict from analyze_background_color
        contrast_mode: 'auto', 'invert', 'enhance', 'outline'
        
    Returns:
        PIL Image: Adjusted logo
    """
    logo_adjusted = logo.copy()
    
    if contrast_mode == 'auto':
        # Choose best method based on background analysis
        if background_analysis['is_light']:
            # Light background - make logo darker
            if background_analysis['is_uniform']:
                contrast_mode = 'darken'
            else:
                contrast_mode = 'outline'
        else:
            # Dark background - make logo lighter
            if background_analysis['is_uniform']:
                contrast_mode = 'lighten'
            else:
                contrast_mode = 'outline'
    
    if contrast_mode == 'invert':
        # Invert logo colors (preserving alpha) - SHARP METHOD
        if logo_adjusted.mode == 'RGBA':
            r, g, b, a = logo_adjusted.split()
            # Use point() method for sharp pixel-perfect inversion
            r = r.point(lambda x: 255 - x)
            g = g.point(lambda x: 255 - x)
            b = b.point(lambda x: 255 - x)
            logo_adjusted = Image.merge('RGBA', (r, g, b, a))
        else:
            logo_adjusted = logo_adjusted.point(lambda x: 255 - x)
    
    elif contrast_mode == 'darken':
        # Make logo darker - PRESERVE SHARPNESS
        if logo_adjusted.mode == 'RGBA':
            r, g, b, a = logo_adjusted.split()
            # Use point() for sharp darkening
            r = r.point(lambda x: int(x * 0.3))
            g = g.point(lambda x: int(x * 0.3))
            b = b.point(lambda x: int(x * 0.3))
            logo_adjusted = Image.merge('RGBA', (r, g, b, a))
        else:
            logo_adjusted = logo_adjusted.point(lambda x: int(x * 0.3))
    
    elif contrast_mode == 'lighten':
        # Make logo lighter - PRESERVE SHARPNESS
        if logo_adjusted.mode == 'RGBA':
            r, g, b, a = logo_adjusted.split()
            # Use point() for sharp lightening
            r = r.point(lambda x: min(255, int(x * 1.8)))
            g = g.point(lambda x: min(255, int(x * 1.8)))
            b = b.point(lambda x: min(255, int(x * 1.8)))
            logo_adjusted = Image.merge('RGBA', (r, g, b, a))
        else:
            logo_adjusted = logo_adjusted.point(lambda x: min(255, int(x * 1.8)))
    
    elif contrast_mode == 'outline':
        # Add outline/border for complex backgrounds - SHARP OUTLINE
        logo_adjusted = add_sharp_outline_to_logo(logo_adjusted, background_analysis)
    
    elif contrast_mode == 'enhance':
        # Enhance contrast - SHARP METHOD
        if logo_adjusted.mode == 'RGBA':
            r, g, b, a = logo_adjusted.split()
            # Sharp contrast enhancement using point()
            r = r.point(lambda x: max(0, min(255, int((x - 128) * 1.5 + 128))))
            g = g.point(lambda x: max(0, min(255, int((x - 128) * 1.5 + 128))))
            b = b.point(lambda x: max(0, min(255, int((x - 128) * 1.5 + 128))))
            logo_adjusted = Image.merge('RGBA', (r, g, b, a))
        else:
            logo_adjusted = logo_adjusted.point(lambda x: max(0, min(255, int((x - 128) * 1.5 + 128))))
    
    return logo_adjusted

def add_sharp_outline_to_logo(logo, background_analysis, outline_width=2):
    """
    Add sharp outline/border to logo for better visibility on complex backgrounds
    OPTIMIZED FOR PIXEL-PERFECT SHARPNESS
    """
    if logo.mode != 'RGBA':
        logo = logo.convert('RGBA')
    
    # Determine outline color based on background
    if background_analysis['is_light']:
        outline_color = (0, 0, 0, 180)  # Dark outline
    else:
        outline_color = (255, 255, 255, 180)  # Light outline
    
    # Get the alpha channel for precise outline creation
    alpha = logo.split()[-1]
    
    # Create outline using morphological operations for sharpness
    outline_mask = Image.new('L', logo.size, 0)
    
    # Create sharp outline by dilating the alpha channel
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            if dx*dx + dy*dy <= outline_width*outline_width:  # Circular outline
                # Shift alpha channel
                shifted_alpha = Image.new('L', logo.size, 0)
                
                # Calculate paste position ensuring it stays within bounds
                paste_x = max(0, dx)
                paste_y = max(0, dy)
                crop_x1 = max(0, -dx)
                crop_y1 = max(0, -dy)
                crop_x2 = logo.width + min(0, -dx)
                crop_y2 = logo.height + min(0, -dy)
                
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    alpha_cropped = alpha.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    shifted_alpha.paste(alpha_cropped, (paste_x, paste_y))
                
                # Combine with existing outline
                outline_mask = Image.blend(outline_mask, shifted_alpha, 0.8)
    
    # Create the outline layer
    outline_img = Image.new('RGBA', logo.size, outline_color)
    outline_img.putalpha(outline_mask)
    
    # Composite outline + original logo using alpha_composite for sharpness
    result = Image.alpha_composite(outline_img, logo)
    
    return result

def resize_logo_sharp(logo, target_size):
    """
    Resize logo while maintaining maximum sharpness
    Uses the best resampling method based on scaling factor
    """
    original_width, original_height = logo.size
    target_width, target_height = target_size
    
    # Calculate scaling factors
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    avg_scale = (width_scale + height_scale) / 2
    
    # Choose optimal resampling method based on scaling
    if avg_scale > 1.0:
        # Upscaling - use LANCZOS for best quality
        resampling = Image.Resampling.LANCZOS
    elif avg_scale > 0.5:
        # Moderate downscaling - use LANCZOS
        resampling = Image.Resampling.LANCZOS
    else:
        # Heavy downscaling - use LANCZOS with pre-filtering
        resampling = Image.Resampling.LANCZOS
    
    # For very small logos, ensure minimum quality
    if target_width < 50 or target_height < 50:
        # Use bicubic for very small sizes to maintain sharpness
        resampling = Image.Resampling.BICUBIC
    
    return logo.resize(target_size, resampling)

def apply_opacity_sharp(logo, opacity):
    """
    Apply opacity while preserving logo sharpness
    Uses point() method for pixel-perfect transparency
    """
    if logo.mode != 'RGBA':
        logo = logo.convert('RGBA')
    
    if opacity >= 1.0:
        return logo
    
    # Split channels
    r, g, b, a = logo.split()
    
    # Apply opacity to alpha channel using point() for sharpness
    alpha_factor = int(opacity * 255) / 255
    a = a.point(lambda x: int(x * alpha_factor))
    
    # Merge channels back
    return Image.merge('RGBA', (r, g, b, a))

def add_watermark(image_path, logo_path, output_path, position='auto', 
                 opacity=0.7, scale='auto', margin='auto', smart_contrast=True):
    """
    Add watermark to an image with resolution-aware sizing, positioning, and smart contrast
    OPTIMIZED FOR MAXIMUM LOGO SHARPNESS
    
    Args:
        image_path: Path to the original image
        logo_path: Path to the PNG logo
        output_path: Path to save the watermarked image
        position: Position of watermark ('auto' for smart positioning, or manual positions)
        opacity: Opacity of the watermark (0.0 to 1.0)
        scale: Scale of the watermark ('auto' for resolution-aware, or float 0.0 to 1.0)
        margin: Margin from edges ('auto' for resolution-aware, or pixels)
        smart_contrast: Enable automatic contrast adjustment based on background
    """
    try:
        # Open the original image - PRESERVE QUALITY
        base_image = Image.open(image_path)
        
        # Convert to RGBA only if necessary
        if base_image.mode != 'RGBA':
            base_image = base_image.convert('RGBA')
        
        # Open the logo - PRESERVE ORIGINAL QUALITY
        logo = Image.open(logo_path)
        
        # Ensure logo is RGBA for transparency handling
        if logo.mode != 'RGBA':
            logo = logo.convert('RGBA')
        
        # Get image dimensions and determine format type
        base_width, base_height = base_image.size
        aspect_ratio = base_width / base_height
        
        # Detect image type based on resolution and aspect ratio
        is_reel = aspect_ratio < 0.7  # Vertical format (9:16 or similar)
        is_4k_horizontal = base_width >= 3840 or (aspect_ratio > 1.5 and base_width >= 1920)
        is_square = 0.9 <= aspect_ratio <= 1.1  # Square format
        is_standard_horizontal = aspect_ratio > 1.1 and not is_4k_horizontal
        
        print(f"  Image: {base_width}x{base_height}, Aspect: {aspect_ratio:.2f}")
        
        # Resolution-aware settings - OPTIMIZED FOR SHARPNESS
        if scale == 'auto':
            if is_reel:
                # Larger watermark for vertical content (more visible)
                watermark_scale = 0.25
                print("  Format: Reel/Vertical")
            elif is_4k_horizontal:
                # Smaller watermark for high-resolution horizontal - but not too small
                watermark_scale = 0.10  # Increased from 0.08 for better visibility
                print("  Format: 4K/High-res Horizontal")
            elif is_square:
                # Medium watermark for square content
                watermark_scale = 0.12
                print("  Format: Square")
            else:
                # Standard horizontal
                watermark_scale = 0.1
                print("  Format: Standard Horizontal")
        else:
            watermark_scale = scale
        
        if margin == 'auto':
            # Adjust margin based on resolution
            if base_width >= 3840:  # 4K+
                margin_px = int(base_width * 0.015)  # 1.5% of width
            elif base_width >= 1920:  # Full HD+
                margin_px = int(base_width * 0.02)   # 2% of width
            elif is_reel:  # Vertical content - smaller margin for better visibility
                margin_px = int(base_width * 0.04)   # 4% of width
            else:
                margin_px = max(20, int(base_width * 0.025))  # 2.5% with minimum
        else:
            margin_px = margin
        
        if position == 'auto':
            # Smart positioning based on format
            if is_reel:
                # For reels, bottom-right works well but not too close to edge
                watermark_position_key = 'bottom-right'
            elif is_square:
                # For square images, bottom-right is standard
                watermark_position_key = 'bottom-right'
            else:
                # For horizontal images, bottom-right is classic
                watermark_position_key = 'bottom-right'
        else:
            watermark_position_key = position
        
        # Calculate watermark size - ENSURE REASONABLE MINIMUM SIZE
        watermark_width = int(base_width * watermark_scale)
        watermark_height = int(logo.height * (watermark_width / logo.width))
        
        # Ensure watermark doesn't exceed reasonable size limits
        max_watermark_size = min(base_width // 3, base_height // 3)
        if watermark_width > max_watermark_size:
            watermark_width = max_watermark_size
            watermark_height = int(logo.height * (watermark_width / logo.width))
        
        # CRITICAL: Ensure minimum size for sharpness
        min_watermark_size = 32  # Minimum 32 pixels
        if watermark_width < min_watermark_size:
            watermark_width = min_watermark_size
            watermark_height = int(logo.height * (watermark_width / logo.width))
            print(f"  Watermark size increased to minimum: {min_watermark_size}px")
        
        print(f"  Watermark: {watermark_width}x{watermark_height}, Margin: {margin_px}px")
        
        # Resize the logo using SHARP method
        logo = resize_logo_sharp(logo, (watermark_width, watermark_height))
        
        # Apply opacity using SHARP method
        logo = apply_opacity_sharp(logo, opacity)
        
        # For debugging - show actual opacity being used
        print(f"  Opacity: {opacity}, Position: {watermark_position_key}")
        
        # Calculate position
        positions = {
            'top-left': (margin_px, margin_px),
            'top-right': (base_width - watermark_width - margin_px, margin_px),
            'bottom-left': (margin_px, base_height - watermark_height - margin_px),
            'bottom-right': (base_width - watermark_width - margin_px, 
                           base_height - watermark_height - margin_px),
            'center': ((base_width - watermark_width) // 2, 
                      (base_height - watermark_height) // 2),
            'bottom-center': ((base_width - watermark_width) // 2,
                            base_height - watermark_height - margin_px)
        }
        
        watermark_position = positions.get(watermark_position_key, positions['bottom-right'])
        print(f"  Final position: {watermark_position}")
        
        # Smart contrast adjustment
        if smart_contrast:
            print("  Analyzing background for contrast optimization...")
            background_analysis = analyze_background_color(base_image, watermark_position, (watermark_width, watermark_height))
            
            avg_r, avg_g, avg_b = background_analysis['avg_color']
            brightness = background_analysis['perceived_brightness']
            bg_type = "light" if background_analysis['is_light'] else "dark"
            uniform = "uniform" if background_analysis['is_uniform'] else "complex"
            
            print(f"  Background: RGB({avg_r},{avg_g},{avg_b}), {bg_type} & {uniform} (brightness: {brightness:.0f})")
            
            # Adjust logo for contrast using SHARP method
            logo = adjust_logo_contrast(logo, background_analysis)
            print(f"  Applied contrast adjustment for {bg_type} {uniform} background")
        
        # Create a transparent overlay with exact positioning
        overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        
        # CRITICAL: Use precise positioning - ensure integer coordinates
        x, y = watermark_position
        x, y = int(round(x)), int(round(y))
        
        # Paste logo with alpha channel for sharp compositing
        overlay.paste(logo, (x, y), logo)
        
        # Composite the images using alpha_composite for maximum sharpness
        watermarked = Image.alpha_composite(base_image, overlay)
        
        # Convert back to RGB if needed (for JPEG output) - PRESERVE QUALITY
        if str(output_path).lower().endswith(('.jpg', '.jpeg')):
            # Convert RGBA to RGB with white background for JPEG
            rgb_image = Image.new('RGB', watermarked.size, (255, 255, 255))
            rgb_image.paste(watermarked, mask=watermarked.split()[-1])
            watermarked = rgb_image
        
        # Save the result with MAXIMUM QUALITY
        save_kwargs = {'quality': 98, 'optimize': False}
        if str(output_path).lower().endswith('.png'):
            save_kwargs = {'optimize': False, 'compress_level': 1}  # Minimal compression for PNG
        
        watermarked.save(output_path, **save_kwargs)
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_directory(input_dir, logo_path, output_dir=None, position='auto',
                     opacity=0.7, scale='auto', margin='auto', overwrite=False, smart_contrast=True):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing images to watermark
        logo_path: Path to the PNG logo
        output_dir: Output directory (if None, creates 'watermarked' subdirectory)
        position: Position of watermark
        opacity: Opacity of the watermark
        scale: Scale of the watermark
        margin: Margin from edges
        overwrite: Whether to overwrite existing files
    """
    input_path = Path(input_dir)
    logo_path = Path(logo_path)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    if not logo_path.exists():
        print(f"Error: Logo file '{logo_path}' does not exist")
        return
    
    # Set output directory
    if output_dir is None:
        output_path = input_path / 'watermarked'
    else:
        output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Process images
    processed = 0
    skipped = 0
    errors = 0
    
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            output_file = output_path / file_path.name
            
            # Skip if file exists and overwrite is False
            if output_file.exists() and not overwrite:
                print(f"Skipping {file_path.name} (already exists)")
                skipped += 1
                continue
            
            print(f"Processing {file_path.name}...")
            
            if add_watermark(file_path, logo_path, output_file, position, 
                           opacity, scale, margin, smart_contrast):
                processed += 1
            else:
                errors += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Errors: {errors} images")
    print(f"Output directory: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Add PNG logo watermark to images with SHARP quality')
    parser.add_argument('input_dir', help='Directory containing images to watermark')
    parser.add_argument('logo', help='Path to PNG logo file')
    parser.add_argument('-o', '--output', help='Output directory (default: input_dir/watermarked)')
    parser.add_argument('-p', '--position', 
                       choices=['auto', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center', 'bottom-center'],
                       default='auto', help='Watermark position (default: auto - smart positioning)')
    parser.add_argument('-a', '--opacity', type=float, default=0.7,
                       help='Watermark opacity 0.0-1.0 (default: 0.7)')
    parser.add_argument('-s', '--scale', default='auto',
                       help='Watermark scale: "auto" for resolution-aware or float 0.0-1.0 (default: auto)')
    parser.add_argument('-m', '--margin', default='auto',
                       help='Margin: "auto" for resolution-aware or pixels (default: auto)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing watermarked images')
    parser.add_argument('--no-smart-contrast', action='store_true',
                       help='Disable automatic contrast adjustment (use original logo colors)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.opacity <= 1.0:
        print("Error: Opacity must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Convert scale and margin if they're not 'auto'
    scale = args.scale
    margin = args.margin
    
    if scale != 'auto':
        try:
            scale = float(scale)
            if not 0.0 <= scale <= 1.0:
                print("Error: Scale must be 'auto' or between 0.0 and 1.0")
                sys.exit(1)
        except ValueError:
            print("Error: Scale must be 'auto' or a number between 0.0 and 1.0")
            sys.exit(1)
    
    if margin != 'auto':
        try:
            margin = int(margin)
            if margin < 0:
                print("Error: Margin must be 'auto' or non-negative")
                sys.exit(1)
        except ValueError:
            print("Error: Margin must be 'auto' or a non-negative integer")
            sys.exit(1)
    
    # Process the directory
    process_directory(
        args.input_dir, 
        args.logo, 
        args.output,
        args.position,
        args.opacity,
        scale,
        margin,
        args.overwrite,
        not args.no_smart_contrast  # Enable smart contrast unless disabled
    )

if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Image Watermark Script - OPTIMIZED FOR SHARP LOGOS")
        print("Usage examples:")
        print("  python watermark.py /path/to/images logo.png")
        print("  python watermark.py /path/to/images logo.png -o /path/to/output")
        print("  python watermark.py /path/to/images logo.png -p center -a 0.5")
        print("  python watermark.py /path/to/images logo.png -s 0.15 -m 50")
        print("  python watermark.py /path/to/images logo.png --no-smart-contrast")
        print("\nSharpness Optimizations:")
        print("  - Uses optimal resampling methods for each scaling scenario")
        print("  - Pixel-perfect opacity and color adjustments")
        print("  - Sharp outline generation for complex backgrounds")
        print("  - Maximum quality save settings")
        print("  - Integer coordinate positioning")
        print("  - Minimum size enforcement")
        print("\nSmart Contrast features:")
        print("  - Analyzes background color where watermark will be placed")
        print("  - Automatically adjusts logo brightness/contrast for visibility")
        print("  - Adds sharp outlines on complex backgrounds")
        print("  - Works with light, dark, and mixed backgrounds")
        print("\nAuto mode intelligently adjusts for:")
        print("  - Reel/Vertical format (9:16, etc.)")
        print("  - 4K/High-resolution horizontal")  
        print("  - Square format (1:1)")
        print("  - Standard horizontal format")
        print("\nUse -h for full help")
        sys.exit(0)
    
    main()