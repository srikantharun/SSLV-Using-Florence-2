#!/usr/bin/env python3
"""
SSLV Rocket Image Analysis Script

This script provides comprehensive image analysis capabilities for Small Satellite Launch Vehicle (SSLV) images.
It includes functionality for:
1. Setting up the environment and loading models
2. Generating synthetic SSLV images using StableDiffusion
3. Analyzing images using Florence-2 multimodal model
4. Segmenting rocket components with computer vision techniques
5. Visualizing rocket components with bounding boxes and masks

Requirements:
- PyTorch
- OpenCV
- Transformers
- Diffusers
- Matplotlib
- PIL
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor
from diffusers import StableDiffusionPipeline

# ===============================
# Configuration and Setup
# ===============================

# Setup device and data types
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Paths
BASE_DIR = "/content/drive/MyDrive/SSLV_Demo"
os.makedirs(BASE_DIR, exist_ok=True)
SSLV_FULL_IMAGE_PATH = os.path.join(BASE_DIR, "sslv_full.png")
SSLV_NOZZLE_IMAGE_PATH = os.path.join(BASE_DIR, "sslv_nozzle.png")
SSLV_ANNOTATED_PATH = os.path.join(BASE_DIR, "sslv_annotated.png")

# Defined rocket components and their colors
COMPONENTS = {
    "nose cone": [177, 0, 332, 85],
    "payload section": [177, 85, 332, 170],
    "rocket nozzle": [177, 409, 332, 512],
    "stabilizing fins": [138, 409, 371, 512]
}

COLORS = {
    "rocket nozzle": (255, 0, 0),    # Red
    "stabilizing fins": (0, 255, 0),  # Green
    "nose cone": (0, 0, 255),         # Blue
    "payload section": (255, 255, 0)  # Yellow
}

# ===============================
# Model Loading Functions
# ===============================

def load_florence_model():
    """Load and initialize the Florence-2 model for image analysis."""
    print("Loading Florence-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", 
        trust_remote_code=True,
        torch_dtype=torch_dtype
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", 
        trust_remote_code=True
    )
    
    return model, processor

def load_stable_diffusion():
    """Load the Stable Diffusion pipeline for image generation."""
    print("Loading Stable Diffusion pipeline...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch_dtype,
            revision="fp16" if torch.cuda.is_available() else "main",
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Optimize memory usage on GPU
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            
        return pipe
    except Exception as e:
        print(f"Error loading Stable Diffusion pipeline: {e}")
        raise

# ===============================
# Image Generation Functions
# ===============================

def generate_rocket_image(pipe, prompt, output_path, height=512, width=512):
    """Generate a synthetic rocket image using Stable Diffusion."""
    print(f"Generating image with prompt: {prompt}")
    try:
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=height,
            width=width
        ).images[0]
        
        # Save the generated image
        image.save(output_path)
        print(f"Image saved to {output_path}")
        
        # Display image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Generated image: {os.path.basename(output_path)}")
        plt.show()
        
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        raise

def generate_sslv_images(pipe):
    """Generate both full SSLV rocket and nozzle closeup images."""
    # Generate full SSLV rocket image
    full_rocket_prompt = (
        "Side view of SSLV rocket on launch pad, showing rocket nozzle, "
        "stabilizing fins, nose cone, payload section. Sharp details, "
        "clean background, labeled structure"
    )
    generate_rocket_image(pipe, full_rocket_prompt, SSLV_FULL_IMAGE_PATH)
    
    # Generate SSLV nozzle closeup
    nozzle_prompt = (
        "A detailed SSLV rocket nozzle in space, high-resolution, realistic, "
        "with metallic texture"
    )
    generate_rocket_image(pipe, nozzle_prompt, SSLV_NOZZLE_IMAGE_PATH)

# ===============================
# Image Analysis Functions
# ===============================

def run_florence_task(model, processor, image, task_prompt, text_input=None):
    """Run a specific task using the Florence-2 model."""
    prompt = task_prompt if text_input is None else task_prompt + text_input
    
    inputs = processor(
        text=prompt, 
        images=image, 
        return_tensors="pt"
    ).to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=False
    )[0]
    
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

def pil_to_cv2(image):
    """Convert PIL Image to OpenCV format."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def analyze_image_with_florence(model, processor, image_path):
    """Perform comprehensive image analysis using Florence-2 model."""
    # Load image
    sslv_image = Image.open(image_path)
    if sslv_image.mode != "RGB":
        sslv_image = sslv_image.convert("RGB")
    
    # Get basic caption
    caption_result = run_florence_task(model, processor, sslv_image, "<CAPTION>")
    if isinstance(caption_result, dict) and "<CAPTION>" in caption_result:
        caption_text = caption_result["<CAPTION>"]
    else:
        caption_text = "SSLV rocket on launch pad"
        print("Warning: Could not get caption, using default text")
    
    # Get detailed caption
    detailed_result = run_florence_task(model, processor, sslv_image, "<DETAILED_CAPTION>")
    if isinstance(detailed_result, dict) and "<DETAILED_CAPTION>" in detailed_result:
        detailed_caption = detailed_result["<DETAILED_CAPTION>"]
    else:
        detailed_caption = "The image shows an SSLV rocket positioned on the launch pad with blue sky in the background."
        print("Warning: Could not get detailed caption, using default text")
    
    # Get object detection results
    detection_result = run_florence_task(model, processor, sslv_image, "<OD>")
    
    # Show the image with caption
    cv_image = pil_to_cv2(sslv_image)
    display_image = cv_image.copy()
    cv2.putText(
        img=display_image,
        text=caption_text,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(255, 255, 255),
        thickness=2
    )
    
    # Display object detection if available
    if isinstance(detection_result, dict) and "<OD>" in detection_result and "bboxes" in detection_result["<OD>"]:
        for box in detection_result["<OD>"]["bboxes"]:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, "Component", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the annotated image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Florence-2 Image Analysis")
    plt.show()
    
    return {
        "image": sslv_image,
        "cv_image": cv_image,
        "caption": caption_text,
        "detailed_caption": detailed_caption,
        "detection": detection_result
    }

# ===============================
# Component Segmentation Functions
# ===============================

def segment_rocket_components(image_path, components=None):
    """Segment rocket components using computer vision techniques."""
    print("Segmenting rocket components...")
    # Load the SSLV rocket image
    full_image = cv2.imread(image_path)
    if full_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Define ROI to include the entire rocket
    x1_roi, y1_roi, x2_roi, y2_roi = 100, 0, 410, 512
    sslv_image = full_image[y1_roi:y2_roi, x1_roi:x2_roi]
    
    # Get dimensions of the cropped image
    height, width, _ = sslv_image.shape
    print(f"Cropped image dimensions: {width}x{height}")
    
    # Use provided components or default ones
    if components is None:
        components = COMPONENTS
    
    # Adjust ROIs for cropped coordinates
    components_cropped = {}
    for component, (x1, y1, x2, y2) in components.items():
        x1_c = x1 - x1_roi
        y1_c = y1 - y1_roi
        x2_c = x2 - x1_roi
        y2_c = y2 - y1_roi
        
        # Validate coordinates
        if x1_c < 0 or x2_c > width or y1_c < 0 or y2_c > height or x1_c >= x2_c or y1_c >= y2_c:
            print(f"Invalid ROI for {component} in cropped image: [{x1_c}, {y1_c}, {x2_c}, {y2_c}]")
            continue
            
        components_cropped[component] = [x1_c, y1_c, x2_c, y2_c]
    
    # Prepare segmentation masks
    masks = {component: np.zeros((height, width), dtype=np.uint8) for component in components_cropped}
    
    # Create initial segmentation masks and refine them using contours
    for component, (x1, y1, x2, y2) in components_cropped.items():
        # Extract the ROI from the image
        roi = sslv_image[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"ROI for {component} is empty, skipping.")
            continue
    
        # Convert ROI to grayscale and apply thresholding to create a binary mask
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No contours found for {component}, using rectangular mask.")
            masks[component][y1:y2, x1:x2] = 255
        else:
            # Create a blank mask for the ROI
            roi_mask = np.zeros_like(thresh)
            cv2.drawContours(roi_mask, contours, -1, 255, thickness=cv2.FILLED)
            # Place the refined mask back into the full mask
            masks[component][y1:y2, x1:x2] = roi_mask
    
        # Draw bounding box on the image (for visualization)
        cv2.rectangle(sslv_image, (x1, y1), (x2, y2), COLORS.get(component, (255, 255, 255)), 2)
        cv2.putText(sslv_image, component, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(component, (255, 255, 255)), 2)
        print(f"Segmentation mask created for: {component} -> [{x1}, {y1}, {x2}, {y2}]")
    
    return {
        "full_image": full_image,
        "cropped_image": sslv_image,
        "masks": masks,
        "components": components_cropped,
        "roi": (x1_roi, y1_roi, x2_roi, y2_roi)
    }

# ===============================
# Visualization Functions
# ===============================

def visualize_segmentation(segmentation_result):
    """Visualize the segmentation of rocket components."""
    full_image = segmentation_result["full_image"]
    sslv_image = segmentation_result["cropped_image"]
    masks = segmentation_result["masks"]
    x1_roi, y1_roi, x2_roi, y2_roi = segmentation_result["roi"]
    
    # Show image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(sslv_image, cv2.COLOR_BGR2RGB))
    plt.title("SSLV Components - Bounding Boxes")
    plt.axis('off')
    plt.show()
    
    # Combine masks into a single RGB segmentation output
    height, width, _ = sslv_image.shape
    seg_mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for component, mask in masks.items():
        seg_mask_rgb[mask > 0] = COLORS.get(component, (255, 255, 255))
    
    # Show combined segmentation mask
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(seg_mask_rgb, cv2.COLOR_BGR2RGB))
    plt.title("SSLV Components - Segmentation Mask")
    plt.axis('off')
    plt.show()
    
    # Map segmentation mask back to the original image with transparency
    full_seg_mask = np.zeros_like(full_image, dtype=np.uint8)
    overlay = full_image.copy()
    alpha = 0.5  # Transparency factor
    
    for component, mask in masks.items():
        # Create a colored mask for the component
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        colored_mask[mask > 0] = COLORS.get(component, (255, 255, 255))
        # Place the colored mask into the full image
        full_seg_mask[y1_roi:y2_roi, x1_roi:x2_roi][mask > 0] = COLORS.get(component, (255, 255, 255))
        # Overlay with transparency
        mask_indices = mask > 0
        overlay[y1_roi:y2_roi, x1_roi:x2_roi][mask_indices] = (
            alpha * colored_mask[mask_indices] + 
            (1 - alpha) * overlay[y1_roi:y2_roi, x1_roi:x2_roi][mask_indices]
        )
    
    # Show the original image with the transparent segmentation overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("SSLV Components - Transparent Overlay")
    plt.axis('off')
    plt.show()
    
    # Show the segmentation mask on the original image (opaque)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(full_seg_mask, cv2.COLOR_BGR2RGB))
    plt.title("SSLV Components - Segmentation Mask (Original Image)")
    plt.axis('off')
    plt.show()

def create_annotation_visualization(image_pil, brief_caption, detailed_caption):
    """Create a comprehensive annotation visualization similar to FLD-5B dataset example."""
    print("Creating annotation visualization...")
    # Get image dimensions for proper scaling
    img_width, img_height = image_pil.size
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4)
    
    # Define title and labels
    fig.suptitle('Dataset FLD-5B with SSLV Rocket Image', fontsize=20)
    
    # Row 1: Image level annotations
    axs[0, 0].text(0.5, 0.5, 'Less granular (image level)', ha='center', va='center', fontsize=12)
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(image_pil)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Image Level', fontsize=14)
    
    axs[0, 2].text(0.5, 0.5, 'More granular (image level)', ha='center', va='center', fontsize=12)
    axs[0, 2].axis('off')
    
    # Row 2: Text annotations
    axs[1, 0].imshow(image_pil)
    axs[1, 0].axis('off')
    axs[1, 0].set_title('None semantic', fontsize=12)
    
    # Create a version with basic bounding box
    axs[1, 1].imshow(image_pil)
    axs[1, 1].add_patch(Rectangle((img_width*0.2, img_height*0.2), img_width*0.6, img_height*0.6,
                                linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.7))
    axs[1, 1].text(img_width*0.2, img_height*0.15, "rocket, launch pad, sky",
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Text annotations', fontsize=14)
    
    # Create a version with detailed annotations
    axs[1, 2].imshow(image_pil)
    axs[1, 2].add_patch(Rectangle((img_width*0.2, img_height*0.2), img_width*0.6, img_height*0.6,
                                linewidth=2, edgecolor='cyan', facecolor='cyan', alpha=0.3))
    axs[1, 2].text(img_width*0.2, img_height*0.15, brief_caption,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    axs[1, 2].axis('off')
    axs[1, 2].set_title('Rich semantic', fontsize=12)
    
    # Row 3: Region-text pairs
    axs[2, 0].imshow(image_pil)
    axs[2, 0].add_patch(Rectangle((img_width*0.25, img_height*0.2), img_width*0.5, img_height*0.6,
                                linewidth=2, edgecolor='yellow', facecolor='none'))
    axs[2, 0].text(img_width*0.25, img_height*0.15, "SSLV rocket on launch pad",
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    axs[2, 0].axis('off')
    axs[2, 0].set_title('Less granular (region level)', fontsize=12)
    
    # Middle plot for row 3
    axs[2, 1].imshow(image_pil)
    axs[2, 1].text(0.5, 0.5, 'Text-phrase-region annotations', ha='center', va='center',
                 transform=axs[2, 1].transAxes, fontsize=14)
    axs[2, 1].axis('off')
    
    # Detailed region annotations
    axs[2, 2].imshow(image_pil)
    axs[2, 2].add_patch(Rectangle((img_width*0.2, img_height*0.2), img_width*0.6, img_height*0.6,
                                linewidth=2, edgecolor='yellow', facecolor='none'))
    
    # Handle long text by breaking it into multiple lines
    text_chunks = [detailed_caption[i:i+40] for i in range(0, len(detailed_caption), 40)]
    wrapped_text = '\n'.join(text_chunks)
    
    axs[2, 2].text(img_width*0.85, img_height*0.25, wrapped_text,
                 fontsize=9, bbox=dict(facecolor='white', alpha=0.7),
                 ha='left', va='top')
    axs[2, 2].axis('off')
    axs[2, 2].set_title('More granular (region level)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the visualization
    plt.savefig(SSLV_ANNOTATED_PATH, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {SSLV_ANNOTATED_PATH}")
    
    return fig

# ===============================
# Main Application Function
# ===============================

def analyze_rocket_image(generate_new_images=False):
    """Main function to perform comprehensive analysis of SSLV rocket images."""
    print("Starting SSLV Rocket Image Analysis...")
    
    # 1. Set up models
    florence_model, florence_processor = load_florence_model()
    
    # 2. Generate images if requested
    if generate_new_images:
        sd_pipe = load_stable_diffusion()
        generate_sslv_images(sd_pipe)
    
    # 3. Analyze image with Florence model
    analysis_results = analyze_image_with_florence(
        florence_model, 
        florence_processor, 
        SSLV_FULL_IMAGE_PATH
    )
    
    # 4. Segment rocket components
    segmentation_results = segment_rocket_components(SSLV_FULL_IMAGE_PATH)
    
    # 5. Visualize segmentation
    visualize_segmentation(segmentation_results)
    
    # 6. Create annotation visualization
    create_annotation_visualization(
        analysis_results["image"],
        analysis_results["caption"],
        analysis_results["detailed_caption"]
    )
    
    print("SSLV Rocket Image Analysis completed successfully!")
    return {
        "analysis": analysis_results,
        "segmentation": segmentation_results
    }

# ===============================
# Application Entry Point
# ===============================

if __name__ == "__main__":
    # Check if GPU is available
    print(f"Using device: {device}")
    print(f"PyTorch dtype: {torch_dtype}")
    
    # Parse command line arguments (if any)
    import argparse
    parser = argparse.ArgumentParser(description='SSLV Rocket Image Analysis')
    parser.add_argument('--generate', action='store_true', help='Generate new rocket images')
    args = parser.parse_args()
    
    # Run the main analysis function
    analyze_rocket_image(generate_new_images=args.generate)
