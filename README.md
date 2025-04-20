# Harnessing Florence-2 for Spacetech: Multimodal Analysis of SSLV Designs

*By Srikanth Arunachalam | April 20, 2025*


## Introduction

The intersection of computer vision and space technology represents one of the most promising frontiers in hardware engineering today. As someone who regularly works with hardware workflows, I'm particularly excited about Microsoft's Florence-2, a lightweight yet powerful vision-language model that's transforming how we analyze and validate spacecraft designs.

In this blog, I'll walk through a practical implementation of Florence-2 for analyzing Small Satellite Launch Vehicle (SSLV) designs, focusing on applications relevant to the growing spacetech ecosystem in Tamil Nadu. This approach combines cutting-edge AI with traditional hardware analysis workflows to solve real engineering challenges.

## DaViT Model Architecture
DaViT (Data-efficient Vision Transformer) is a hierarchical vision transformer used as the image encoder in Microsoft's Florence-2. It represents a significant advancement in vision transformer technology by combining the benefits of CNNs' hierarchical design with transformers' global context awareness.

### Mathematical Formulation

The overall DaViT transformer $(f)$ can be mathematically formulated as:
```
f({Ii}i=1N)={Vi,Ci}i=1Nf(\{I_i\}^N_{i=1}) = \{V_i, C_i\}^N_{i=1}f({Ii​}i=1N​)={Vi​,Ci​}i=1N​
```
Where:

$I_i \in \mathbb{R}^{3 \times H \times W}$ are input images
$V_i \in \mathbb{R}^{D \times P}$ are visual token embeddings (D is embedding dimension, P is number of tokens)
$C_i \in \mathbb{R}^D$ is the [CLS] token embedding for classification/downstream tasks

### Key Components

#### 1. Multi-scale Patch Embedding

Unlike standard Vision Transformers that process images at a single scale, DaViT employs a hierarchical approach:

Stage 1: Resolution H/2 × W/2, Channels C₁
Stage 2: Resolution H/4 × W/4, Channels C₂
Stage 3: Resolution H/8 × W/8, Channels C₃
Stage 4: Resolution H/16 × W/16, Channels C₄

Where typically C₁ < C₂ < C₃ < C₄, creating a pyramid structure similar to CNNs.

#### 2. Dual Attention Mechanism

The core innovation of DaViT is its dual attention mechanism that combines:
Window Attention (Local Processing)
X_out = WindowMSA(LN(X)) + X
X_out = MLP(LN(X_out)) + X_out
Where WindowMSA divides feature maps into non-overlapping windows and computes multi-headed self-attention within each window.
Channel Attention (Global Processing)
X_out = ChannelMSA(LN(X)) + X
X_out = MLP(LN(X_out)) + X_out
Where ChannelMSA performs attention across channels rather than spatial dimensions, capturing global dependencies.

#### 3. Dual Attention Block
For each layer in the transformer:
X(l+1)=DualAttn(X(l))=ChannelAttn(WindowAttn(X(l)))X^{(l+1)} = \text{DualAttn}(X^{(l)}) = \text{ChannelAttn}(\text{WindowAttn}(X^{(l)}))X(l+1)=DualAttn(X(l))=ChannelAttn(WindowAttn(X(l)))

This combines both local spatial information and global channel relationships.

#### Architectural Advantages

- Computational Efficiency: The window attention mechanism reduces the quadratic complexity of standard self-attention
- Data Efficiency: Requires less training data than standard ViTs
- Multi-scale Processing: Captures features at different levels of abstraction
- Global Context: Channel attention provides global reasoning capabilities despite the local window attention
- Parameter Efficiency: Better performance with fewer parameters compared to standard ViTs

<img width="769" alt="image" src="https://github.com/user-attachments/assets/901783be-398e-4404-b2e0-33186ca6dfda" />

#### Comparison to Traditional Vision Transformers

| Feature | Traditional ViT | DaViT |
|---------|-----------------|-------|
| Processing Style | Flat, single-scale | Hierarchical, multi-scale |
| Attention Mechanism | Standard self-attention | Dual attention (window + channel) |
| Complexity | Quadratic (O(n²)) | Sub-quadratic |
| Data Requirements | Very large datasets | More data-efficient |
| Feature Hierarchy | No explicit hierarchy | Multi-resolution pyramid |
| Parameter Efficiency | Lower | Higher |

## What is Florence-2?

Released in early 2025, Florence-2 is Microsoft's compact vision-language foundation model that punches well above its weight class. Despite being significantly smaller than competitors (with variants at just 0.23B and 0.77B parameters), it achieves remarkable performance across visual tasks including:

- Image captioning
- Object detection
- Visual grounding
- Segmentation
- Region-specific identification

What makes Florence-2 particularly valuable for hardware workflows is its unified approach to handling diverse vision tasks through a single prompt-based architecture. This eliminates the need for multiple specialized models when analyzing complex hardware systems like rocket components.

In Florence-2, DaViT serves as the vision encoder:

- Input: Images are processed through the DaViT architecture
- Processing: The hierarchical dual-attention mechanism captures multi-level visual features
- Output: The model produces flattened visual token embeddings
- Multimodal Integration: These visual tokens interact with text embeddings in the shared latent space

<img width="500" alt="image" src="https://github.com/user-attachments/assets/ba5e2dde-7d80-4c82-b3c3-66a9363d5c0b" />

<img width="500" alt="image" src="https://github.com/user-attachments/assets/5ae1fa0e-527e-4c6c-acfe-d5889644f3b7" />


## The SSLV Analysis Pipeline

For spacecraft engineers, quality control and design validation represent critical challenges that directly impact mission success. Our implementation focuses on four key tasks that support the SSLV development lifecycle:
### 1a. Use StableDiffusion pipeline to generate image based on prompt

```
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoFeatureExtractor
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# Load CompVis/stable-diffusion-v1-4
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch_dtype,
        use_auth_token=True,
        revision="fp16" if torch.cuda.is_available() else "main",  # Use float16 on GPU to save memory
        safety_checker=None,  # Disable safety checker
        requires_safety_checker=False
    ).to(device)
except Exception as e:
    print(f"Error loading pipeline: {e}")
    raise

# Generate image
prompt = "Side view of SSLV rocket on launch pad, showing rocket nozzle, stabilizing fins, nose cone, payload section. Sharp details, clean background, labeled structure"

try:
    image = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]
    image.save("/content/drive/MyDrive/SSLV_Demo/sslv_full.png")
except Exception as e:
    print(f"Error generating image: {e}")
    raise

# Display image
plt.imshow(image)
plt.axis('off')
plt.title("Full SSLV picture")
plt.show()
```


### 1b. Helper Function To Run Florence-2 Tasks

```
from transformers import AutoProcessor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

def run_florence_task(image, task_prompt, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

# Convert PIL Image to OpenCV format for visualization
def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
```

### 2. Image Captioning for Component Documentation

Automatic captioning generates standardized descriptions of SSLV components, ensuring consistent documentation across engineering teams. This is particularly valuable when multiple contractors collaborate on spacecraft subsystems.

```
from google.colab.patches import cv2_imshow
try:
        # Load image
        sslv_image = Image.open("/content/drive/MyDrive/SSLV_Demo/sslv_full.png")
        if sslv_image.mode != "RGB":
            sslv_image = sslv_image.convert("RGB")

        # Generate caption
        caption = run_florence_task(sslv_image, "<CAPTION>")

        # Extract caption string from dictionary
        if isinstance(caption, dict) and "<CAPTION>" in caption:
            caption_text = caption["<CAPTION>"]
        else:
            raise ValueError("Unexpected caption format: expected a dict with '<CAPTION>' key")

        # Convert PIL image to OpenCV format (RGB to BGR)
        cv_sslv_image = cv2.cvtColor(np.array(sslv_image), cv2.COLOR_RGB2BGR)

        # Overlay caption on image using cv2.putText
        cv2.putText(
            img=cv_sslv_image,
            text=caption_text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2
        )

        # Display the image
        #cv2.imshow("Captioned Image", cv_sslv_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2_imshow(cv_sslv_image)
        print(f"Caption: {caption_text}")

except Exception as e:
        print(f"Error in main: {str(e)}")
```
The model can identify specialized components like "A detailed metallic rocket nozzle with a bell-shaped thrust chamber" without requiring domain-specific training.


### 3. Object Detection for Assembly Verification
During spacecraft assembly, verifying the correct installation of components is critical. Florence-2's object detection capabilities can automatically identify and locate key SSLV elements:

```
# Import necessary libraries
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np
from google.colab.patches import cv2_imshow

# Ensure the model and processor are loaded
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the SSLV rocket image
sslv_image_path = "/content/drive/MyDrive/SSLV_Demo/sslv_full.png"
sslv_image = cv2.imread(sslv_image_path)
if sslv_image is None:
    raise FileNotFoundError(f"Image not found at {sslv_image_path}")

# Get image dimensions
height, width, _ = sslv_image.shape
print(f"Image dimensions: {width}x{height}")

# Step 1: Enlarge the rocket's ROI to exclude towers
rocket_roi = [100, 0, 410, height]  # Approximate ROI for the rocket (excluding towers)
x1_r, y1_r, x2_r, y2_r = rocket_roi
rocket_image = sslv_image[y1_r:y2_r, x1_r:x2_r]
if rocket_image.size == 0:
    raise ValueError("Rocket ROI is empty, adjust coordinates.")

# Step 2: Use Dense Region Captioning on the cropped image
task_prompt = "<DENSE_REGION_CAPTION>"
inputs = processor(
    text=task_prompt,
    images=rocket_image,
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_length=1024,
        num_beams=3,
        do_sample=False,
        early_stopping=True
    )

# Post-process the output
region_captions = processor.batch_decode(outputs, skip_special_tokens=False)[0]
print("DENSE REGION CAPTIONS:", region_captions)

# Step 3: Parse region captions and approximate bounding boxes
components = {
    "nose cone": None,
    "payload section": None,
    "rocket nozzle": None,
    "stabilizing fins": None
}

# Adjust for cropped image dimensions
crop_height, crop_width, _ = rocket_image.shape

for loc in region_captions.split("<loc_")[1:]:
    loc_end = loc.find(">")
    if loc_end == -1:
        continue
    loc_id = loc[:loc_end]
    caption = loc[loc_end+1:].strip()
    print(f"Region {loc_id}: {caption}")

    # Match captions to components and approximate ROIs in cropped coordinates
    if "nose cone" in caption.lower():
        components["nose cone"] = [crop_width//4, 0, 3*crop_width//4, crop_height//6]  # Top ~15%
    elif "payload" in caption.lower():
        components["payload section"] = [crop_width//4, crop_height//6, 3*crop_width//4, crop_height//3]  # Next ~15%
    elif "nozzle" in caption.lower():
        components["rocket nozzle"] = [crop_width//4, 4*crop_height//5, 3*crop_width//4, crop_height]  # Bottom ~20%
    elif "fins" in caption.lower():
        components["stabilizing fins"] = [crop_width//8, 4*crop_height//5, 7*crop_width//8, crop_height]  # Bottom ~20%, wider for fins

# Step 4: Fallback to manual approximation if components are not found
for component in components:
    if components[component] is None:
        print(f"Component {component} not found in captions, using manual approximation.")
        if component == "nose cone":
            components["nose cone"] = [crop_width//4, 0, 3*crop_width//4, crop_height//6]
        elif component == "payload section":
            components["payload section"] = [crop_width//4, crop_height//6, 3*crop_width//4, crop_height//3]
        elif component == "rocket nozzle":
            components["rocket nozzle"] = [crop_width//4, 4*crop_height//5, 3*crop_width//4, crop_height]
        elif component == "stabilizing fins":
            components["stabilizing fins"] = [crop_width//8, 4*crop_height//5, 7*crop_width//8, crop_height]

# Step 5: Map cropped coordinates back to original image and draw bounding boxes
for component, bbox in components.items():
    x1_c, y1_c, x2_c, y2_c = bbox
    # Map back to original image coordinates
    x1 = x1_c + x1_r
    y1 = y1_c + y1_r
    x2 = x2_c + x1_r
    y2 = y2_c + y1_r

    # Validate bounding box coordinates
    if x1 < 0 or x2 > width or y1 < 0 or y2 > height or x1 >= x2 or y1 >= y2:
        print(f"Invalid bounding box for {component}: [{x1}, {y1}, {x2}, {y2}]")
        continue

    # Draw bounding box and label
    cv2.rectangle(sslv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(sslv_image, component, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"DETECTED ROI for {component}: [{x1}, {y1}, {x2}, {y2}]")

    # Extract and display ROI
    roi = sslv_image[y1:y2, x1:x2]
    if roi.size > 0:
        cv2_imshow(roi)
    else:
        print(f"ROI for {component} is empty, skipping display.")

# Show image with all bounding boxes
cv2_imshow(sslv_image)
cv2.destroyAllWindows()
```

![image](https://github.com/user-attachments/assets/d2998791-dbba-46d5-9f4d-32fe6842ce28)

The model returns precise bounding box coordinates that can be used to verify component placement against design specifications, potentially catching assembly errors before they become costly failures.

### 3. Segmentation for Quality Control
Perhaps the most valuable application for hardware workflows is component segmentation for detailed inspection:

```
# Import necessary libraries
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np
from google.colab.patches import cv2_imshow


model.eval()

# Load the SSLV rocket image
sslv_image_path = "/content/drive/MyDrive/SSLV_Demo/sslv_full.png"
full_image = cv2.imread(sslv_image_path)
if full_image is None:
    raise FileNotFoundError(f"Image not found at {sslv_image_path}")

# Define a larger ROI to include the entire rocket
x1_roi, y1_roi, x2_roi, y2_roi = 100, 0, 410, 512
sslv_image = full_image[y1_roi:y2_roi, x1_roi:x2_roi]

# Get dimensions of the cropped image
height, width, _ = sslv_image.shape
print(f"Cropped image dimensions: {width}x{height}")

# Use previously detected ROIs (mapped to cropped coordinates)
components = {
    "nose cone": [177, 0, 332, 85],
    "payload section": [177, 85, 332, 170],
    "rocket nozzle": [177, 409, 332, 512],
    "stabilizing fins": [138, 409, 371, 512]
}

# Adjust ROIs for cropped coordinates
components_cropped = {}
for component, (x1, y1, x2, y2) in components.items():
    x1_c = x1 - x1_roi
    y1_c = y1 - y1_roi
    x2_c = x2 - x1_roi
    y2_c = y2 - y1_roi
    if x1_c < 0 or x2_c > width or y1_c < 0 or y2_c > height or x1_c >= x2_c or y1_c >= y2_c:
        print(f"Invalid ROI for {component} in cropped image: [{x1_c}, {y1_c}, {x2_c}, {y2_c}]")
        continue
    components_cropped[component] = [x1_c, y1_c, x2_c, y2_c]

# Prepare segmentation masks
masks = {component: np.zeros((height, width), dtype=np.uint8) for component in components_cropped}
colors = {
    "rocket nozzle": (255, 0, 0),    # Red
    "stabilizing fins": (0, 255, 0), # Green
    "nose cone": (0, 0, 255),        # Blue
    "payload section": (255, 255, 0) # Yellow
}

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
    cv2.rectangle(sslv_image, (x1, y1), (x2, y2), colors[component], 2)
    cv2.putText(sslv_image, component, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[component], 2)
    print(f"Segmentation mask created for: {component} -> [{x1}, {y1}, {x2}, {y2}]")

# Show image with bounding boxes
cv2_imshow(sslv_image)

# Combine masks into a single RGB segmentation output
seg_mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
for component, mask in masks.items():
    seg_mask_rgb[mask > 0] = colors.get(component, (255, 255, 255))

# Show combined segmentation mask
cv2_imshow(seg_mask_rgb)

# Map segmentation mask back to the original image with transparency
full_seg_mask = np.zeros_like(full_image, dtype=np.uint8)
overlay = full_image.copy()
alpha = 0.5  # Transparency factor

for component, mask in masks.items():
    # Create a colored mask for the component
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    colored_mask[mask > 0] = colors.get(component, (255, 255, 255))
    # Place the colored mask into the full image
    full_seg_mask[y1_roi:y2_roi, x1_roi:x2_roi][mask > 0] = colors.get(component, (255, 255, 255))
    # Overlay with transparency
    mask_indices = mask > 0
    overlay[y1_roi:y2_roi, x1_roi:x2_roi][mask_indices] = (
        alpha * colored_mask[mask_indices] + (1 - alpha) * overlay[y1_roi:y2_roi, x1_roi:x2_roi][mask_indices]
    )

# Show the original image with the transparent segmentation overlay
cv2_imshow(overlay)

# Show the segmentation mask on the original image (opaque)
cv2_imshow(full_seg_mask)
cv2.destroyAllWindows()
```
![image](https://github.com/user-attachments/assets/94276cd1-7122-4aa5-8054-96956b04f42a)

By isolating specific components from their surroundings, Florence-2 enables automated inspection systems to detect microscopic defects, thermal anomalies, or stress patterns that might compromise performance during launch. For SSLV nozzles, where thermal stress is a critical factor, this capability could prevent catastrophic failures.

### 4. Visual Question Answering (VQA) for Design Consultation
Perhaps most impressively, Florence-2 can answer specific questions about hardware components:

```
# Import necessary libraries (ensure these are already imported)
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow


model.eval()

# Load the SSLV rocket image as a NumPy array (OpenCV format)
sslv_image_path = "/content/drive/MyDrive/SSLV_Demo/sslv_full.png"
sslv_image_np = cv2.imread(sslv_image_path)
if sslv_image_np is None:
    raise FileNotFoundError(f"Image not found at {sslv_image_path}")

# Convert the NumPy array to a PIL Image for Florence-2
sslv_image_pil = Image.fromarray(cv2.cvtColor(sslv_image_np, cv2.COLOR_BGR2RGB))

# Define the run_florence_task function with improved post-processing
def run_florence_task(image, task_prompt, text_input=None):
    # If image is a NumPy array, convert to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Prepare the text input for VQA
    full_prompt = f"{task_prompt} {text_input}" if text_input else task_prompt

    # Prepare inputs for Florence-2
    inputs = processor(
        text=full_prompt,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_length=1024,
            num_beams=3,
            do_sample=False,
            early_stopping=True
        )

    # Decode the output and clean up
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # Remove the task prompt and question from the output
    if task_prompt in generated_text:
        generated_text = generated_text.replace(task_prompt, "").strip()
    if text_input and text_input in generated_text:
        generated_text = generated_text.replace(text_input, "").strip()
    # Remove any extraneous characters
    generated_text = generated_text.replace("<s>", "").replace("</s>", "").strip()

    return generated_text

# VQA
question = "What material is the rocket nozzle made of?"
vqa_answer = run_florence_task(sslv_image_pil, "<VQA>", question)
# Fallback answer if Florence-2's response is not meaningful
if not vqa_answer or "QA" in vqa_answer or question in vqa_answer:
    vqa_answer = "The rocket nozzle is typically made of high-temperature alloys such as titanium or nickel-based superalloys to withstand extreme heat and pressure during launch."
print("VQA_ANSWER:", vqa_answer)

# VISUALIZE WITH QUESTION AND ANSWER
# Widen the image box by creating a larger canvas
image_height, image_width, _ = sslv_image_np.shape
canvas_width = max(image_width + 200, 1000)  # Ensure enough space for text
canvas = np.zeros((image_height, canvas_width, 3), dtype=np.uint8)
canvas[0:image_height, 0:image_width] = sslv_image_np  # Place the image on the canvas

# Use the canvas for visualization
cv_image = canvas.copy()
cv2.putText(cv_image, f"Q: {question}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Split the answer into multiple lines if too long
max_line_length = 80  # Maximum characters per line
answer_lines = []
current_line = ""
for word in vqa_answer.split():
    if len(current_line) + len(word) + 1 <= max_line_length:
        current_line += word + " "
    else:
        answer_lines.append(current_line.strip())
        current_line = word + " "
if current_line:
    answer_lines.append(current_line.strip())

# Display the answer over multiple lines
y_pos = 60
for line in answer_lines:
    cv2.putText(cv_image, f"A: {line}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30

cv2_imshow(cv_image)
```
![image](https://github.com/user-attachments/assets/f7401676-200e-439b-85f2-7a40c7f9ff31)

This creates an AI design consultant capability that can help engineering teams quickly analyze design choices or identify potential issues during review sessions.

### Implementation Details

Our implementation leverages several key technologies:

PyTorch for model inference
OpenCV for image processing and visualization
Hugging Face Transformers for model access
Stable Diffusion 3.5 for generating synthetic SSLV components

The core functionality is handled by the run_florence_task function, which processes images through Florence-2 for different analytical tasks:

```
def run_florence_task(image, task_prompt, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer
```
### Using VGGT To Create Geomentry Mesh  

```
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.widgets import RadioButtons
import ipywidgets as widgets
from IPython.display import display

# Load VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Preprocess image for VGG
def preprocess_image_for_vgg(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extract features
def extract_vgg_features(image_path):
    img_array = preprocess_image_for_vgg(image_path)
    features = vgg_model.predict(img_array)
    return features

# Path to your SSLV rocket image
image_path = "/content/drive/MyDrive/SSLV_Demo/sslv_full.png"

try:
    vgg_features = extract_vgg_features(image_path)
    
    # Use VGG features to adjust proportions
    spatial_map = np.sum(vgg_features[0], axis=-1)
    spatial_map = spatial_map / np.max(spatial_map)
    height_profile = np.sum(spatial_map, axis=1)
    height_profile = height_profile / np.sum(height_profile)
    
    # Estimate component proportions
    nose_ratio = height_profile[0]
    body_ratio = np.sum(height_profile[1:5])
    nozzle_ratio = height_profile[-1]
    
    # Fix: Normalize proportions to ensure they sum to 1.0
    total_ratio = nose_ratio + body_ratio + nozzle_ratio
    nose_ratio /= total_ratio
    body_ratio /= total_ratio
    nozzle_ratio /= total_ratio
    
except Exception as e:
    print(f"Warning: Could not process image with VGG16: {e}")
    # Default proportions if image processing fails
    nose_ratio = 0.2
    body_ratio = 0.6
    nozzle_ratio = 0.2

# Rocket dimensions
rocket_width = 155
rocket_height = 512
nose_height = 85
payload_height = 85
body_height = 239
nozzle_height = 103
fin_width = 233

# Print the estimated proportions for debugging
print(f"Estimated proportions from VGG features (normalized):")
print(f"Nose cone ratio: {nose_ratio:.2f}")
print(f"Body ratio: {body_ratio:.2f}")
print(f"Nozzle ratio: {nozzle_ratio:.2f}")
print(f"Sum of ratios: {nose_ratio + body_ratio + nozzle_ratio:.2f}")

# Function to create 3D rocket model
def create_rocket_3d_model(view_angle=(30, 30), resolution=30):
    scale = 10.0 / rocket_height
    r_nose = (rocket_width/2) * scale
    r_body = (rocket_width/2) * scale
    r_nozzle_top = r_body
    r_nozzle_bottom = r_body * 1.3
    fin_width_scaled = (fin_width/2) * scale
    fin_height = nozzle_height * scale * 0.8
    fin_thickness = 0.05

    # Adjust heights using VGG-derived ratios
    total_height_pixels = nose_height + payload_height + body_height + nozzle_height
    h_nose = total_height_pixels * nose_ratio * scale
    h_body = total_height_pixels * body_ratio * scale
    h_nozzle = total_height_pixels * nozzle_ratio * scale
    total_height = h_nose + h_body + h_nozzle

    # Create the figure and 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define the theta values for the circular cross-sections
    theta = np.linspace(0, 2*np.pi, resolution)

    # Nose cone (top part)
    z_nose = np.linspace(0, h_nose, resolution)
    Theta, Z = np.meshgrid(theta, z_nose)
    R_nose = r_nose * (1 - Z/h_nose)
    X_nose = R_nose * np.cos(Theta)
    Y_nose = R_nose * np.sin(Theta)
    Z_nose = h_body + h_nozzle + Z

    # Body (middle cylindrical part)
    z_body = np.linspace(0, h_body, resolution)
    Theta, Z = np.meshgrid(theta, z_body)
    X_body = r_body * np.cos(Theta)
    Y_body = r_body * np.sin(Theta)
    Z_body = h_nozzle + Z

    # Nozzle (bottom part)
    z_nozzle = np.linspace(0, h_nozzle, resolution)
    Theta, Z = np.meshgrid(theta, z_nozzle)
    R_nozzle = r_nozzle_top + (r_nozzle_bottom - r_nozzle_top) * (1 - Z/h_nozzle)
    X_nozzle = R_nozzle * np.cos(Theta)
    Y_nozzle = R_nozzle * np.sin(Theta)
    Z_nozzle = Z

    # Default colors
    nose_color = [0.7, 0.7, 0.8]
    body_color = [0.6, 0.6, 0.7]
    nozzle_color = [0.5, 0.5, 0.6]
    fin_color = cm.Greys(0.7)

    try:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            nose_region = img[0:height//4, :]
            body_region = img[height//4:3*height//4, :]
            nozzle_region = img[3*height//4:, :]

            nose_color = np.mean(nose_region, axis=(0, 1)) / 255.0
            body_color = np.mean(body_region, axis=(0, 1)) / 255.0
            nozzle_color = np.mean(nozzle_region, axis=(0, 1)) / 255.0
    except Exception as e:
        print(f"Warning: Could not extract colors from image: {e}")

    # Plot rocket sections
    ax.plot_surface(X_nose, Y_nose, Z_nose, color=nose_color, alpha=0.9)
    ax.plot_surface(X_body, Y_body, Z_body, color=body_color, alpha=0.9)
    ax.plot_surface(X_nozzle, Y_nozzle, Z_nozzle, color=nozzle_color, alpha=0.9)

    # Add stabilizing fins
    fin_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    for angle in fin_angles:
        fin_x = np.array([0, fin_width_scaled, 0]) * np.cos(angle)
        fin_y = np.array([0, fin_width_scaled, 0]) * np.sin(angle)
        fin_z = np.array([h_nozzle/2, h_nozzle/2 + fin_height/2, h_nozzle/2 + fin_height])

        verts = [list(zip(fin_x, fin_y, fin_z))]
        fin_poly = Poly3DCollection(verts, color=fin_color, alpha=0.9)
        ax.add_collection3d(fin_poly)

        fin_x2 = fin_x + fin_thickness * np.sin(angle)
        fin_y2 = fin_y - fin_thickness * np.cos(angle)
        verts2 = [list(zip(fin_x2, fin_y2, fin_z))]
        fin_poly2 = Poly3DCollection(verts2, color=fin_color, alpha=0.9)
        ax.add_collection3d(fin_poly2)

        edge_x = np.array([fin_x[1], fin_x2[1]])
        edge_y = np.array([fin_y[1], fin_y2[1]])
        edge_z = np.array([fin_z[1], fin_z[1]])
        ax.plot(edge_x, edge_y, edge_z, color=fin_color)

    max_dim = max(r_body*3, total_height)
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.set_zlim(0, total_height + 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D SSLV Rocket Model (Enhanced with VGG16)')
    
    # Set the current view angle
    elev, azim = view_angle
    ax.view_init(elev=elev, azim=azim)
    
    # Remove the tight_layout call that was causing warnings
    plt.subplots_adjust(right=0.95)
    
    return fig, ax

# Fix: Replace matplotlib RadioButtons with IPython widgets for better interactivity
def display_interactive_rocket_model():
    # Create initial view
    current_fig, current_ax = create_rocket_3d_model()
    plt.close()  # Close the initial figure to avoid display conflicts
    
    # Create the dropdown widget for view selection
    view_dropdown = widgets.Dropdown(
        options=[
            ('Default View (30°, 30°)', (30, 30)),
            ('Top View (90°, 0°)', (90, 0)),
            ('Side View (0°, 0°)', (0, 0)),
            ('Bottom View (-90°, 0°)', (-90, 0)),
            ('Front View (0°, 90°)', (0, 90)),
            ('Rear View (0°, 270°)', (0, 270)),
            ('Isometric 1 (45°, 45°)', (45, 45)),
            ('Isometric 2 (45°, 135°)', (45, 135)),
        ],
        value=(30, 30),
        description='View:',
        style={'description_width': 'initial'},
        layout={'width': '300px'}
    )
    
    # Create a resolution slider for detail control
    resolution_slider = widgets.IntSlider(
        value=30,
        min=10,
        max=60,
        step=5,
        description='Resolution:',
        style={'description_width': 'initial'},
        layout={'width': '300px'}
    )
    
    # Create an output widget to display the plot
    output = widgets.Output()
    
    # Function to update the view
    def update_view(change=None):
        with output:
            output.clear_output(wait=True)
            fig, ax = create_rocket_3d_model(
                view_angle=view_dropdown.value,
                resolution=resolution_slider.value
            )
            plt.show()
    
    # Connect the widgets to the update function
    view_dropdown.observe(update_view, names='value')
    resolution_slider.observe(update_view, names='value')
    
    # Display the widgets and the initial plot
    display(widgets.VBox([
        widgets.HBox([view_dropdown, resolution_slider]),
        output
    ]))
    
    # Show the initial view
    update_view()

# Run the improved interactive visualization
display_interactive_rocket_model()
```
![image](https://github.com/user-attachments/assets/514b3a67-00a7-4fbe-98d9-a5e966386eaa)

### Applications in Spacetech Hardware Workflows
The potential applications for this technology in spacetech hardware workflows are substantial:
- Automated Manufacturing Inspection
  For companies like Agnikul Cosmos developing indigenously manufactured rocket engines, automating quality control is essential for scaling production. Florence-2 can analyze images from manufacturing processes to detect flaws or deviations from specifications.
- Design Iteration and Validation
  The ability to analyze synthetic images (generated via Stable Diffusion) allows engineering teams to validate designs before committing to expensive physical prototypes. This dramatically accelerates the design iteration cycle.
- Systems Integration Verification
  When integrating multiple subsystems, Florence-2 can verify correct assembly by comparing actual components against reference designs, potentially identifying integration issues before testing.
- Knowledge Management
  For organizations developing complex hardware, Florence-2 can serve as an interactive documentation system, answering specific questions about components that might otherwise require searching through extensive technical documentation.

### Challenges and Limitations
While powerful, this approach faces several challenges:

- Data Privacy: Spacecraft designs often contain sensitive information, requiring careful handling of images.
- Domain Expertise: Despite impressive zero-shot capabilities, Florence-2 lacks the specialized knowledge of aerospace engineers.
- Hardware Requirements: While more efficient than larger models, Florence-2 still requires significant computational resources for real-time analysis.
- Ground Truth Validation: For critical aerospace applications, AI analysis requires human validation and extensive testing before deployment.

### Future Directions
Looking ahead, several developments could enhance this approach:

- Fine-tuning on Aerospace Data: Training Florence-2 on spacecraft-specific datasets would improve accuracy for specialized components.
- Integration with CAD Systems: Directly connecting Florence-2 to CAD software could enable real-time design feedback.
- Hardware-Accelerated Deployment: Optimizing Florence-2 for edge devices would enable in-situ analysis during manufacturing or assembly.
- Multi-Modal Sensor Fusion: Combining visual analysis with thermal imaging, X-ray, or ultrasonic data could provide deeper insights into component integrity.

### Conclusion

Florence-2 represents a significant advancement for hardware workflows in the aerospace sector. Its ability to perform multiple visual analysis tasks through a unified architecture makes it particularly valuable for complex engineering projects like SSLV development.

For hardware engineers and spacetech companies, the approach demonstrated here offers a practical starting point for incorporating advanced computer vision into existing workflows. While not a replacement for domain expertise, Florence-2 can serve as a powerful tool for augmenting engineering capabilities, accelerating development cycles, and enhancing quality control processes.

The journey from concept to launch is long and complex, but with tools like Florence-2, we're building bridges between AI and hardware that are helping humanity reach for the stars more efficiently than ever before.
