# Detecting_AI_Images_CAP320

## Step 1: Choose a Media Type
I will be using images.

## Step 2: Collect Examples
I labeled the human generated images from subreddit r/dndcommissions (https://www.reddit.com/r/dndcommissions/) as:
  - human_ex1
  - human_ex2
  - human_ex3

I saved the AI generated images from microsoft Copilot as:
  - AI_ex1
  - AI_ex2
  - AI_ex3

## Step 3: Brainstorm Specific Traits with AI
AI Response:
```
Detecting whether an image is AI-generated often involves identifying specific traits and features that distinguish it from human-created content. These indicators may include visual inconsistencies, artifacts, and peculiar patterns. Here's a comprehensive list to help you identify AI-generated images:

1. Anatomical Inaccuracies
Hands and Fingers: AI often struggles with hand anatomy, leading to too many or too few fingers, misshapen palms, or unnatural finger bends.
Facial Features: Look for misaligned eyes, distorted teeth, or asymmetrical facial elements.
Body Proportions: Odd body shapes, unrealistic poses, or fused body parts.
2. Text and Typography Artifacts
AI-generated images often fail to produce coherent text, resulting in gibberish letters or oddly shaped fonts.
3. Background and Context Clues
Inconsistent Details: Objects in the background may blur together or exhibit strange distortions.
Unnatural Transitions: Look for abrupt or poorly blended changes in textures (e.g., sky merging with water unnaturally).
4. Repetition and Pattern Artifacts
AI can produce repetitive patterns where they shouldn't exist, such as duplicate leaves, flowers, or tiles.
Textures like hair or grass might appear unnaturally uniform or chaotic.
5. Lighting and Shadows
Inconsistent Shadows: Shadows may not match light sources or be missing entirely.
Unnatural Highlights: Uneven distribution of light and unrealistic reflections.
6. Surface Artifacts
Blurry regions alongside overly sharp details.
Smudges or pixel-like artifacts in areas requiring fine detail.
7. Color and Texture Issues
Overly smooth or "plastic" skin textures.
Inconsistent coloring within natural objects (like blending errors in fabric).
8. Emotional "Uncanniness"
AI-generated faces can evoke a sense of something being slightly "off" despite appearing photorealistic.
Tools for Detection:
Forensic Tools: Programs like FotoForensics analyze noise inconsistencies and compression artifacts.
Reverse Image Search: Helps check whether a similar image exists online.
AI Detection Software: Tools like Deepware Scanner or DARPA's SemaFor can help identify manipulated content.
Would you like suggestions on any detection tools or further examples?
```

The 3 that I chose to work with were:
1. Surface Artifacts
2. Anatomical Inaccuracies
3. Background and Context Clues

## Step 4: Prompt an AI to Write a Detection Program
My Prompt:
```
Create a Python program to determine if an image is AI-generated or human-generated. The program should analyze the following traits: Surface Artifacts, Anatomical Inaccuracies, as well as Background and Context Clues. Use image processing libraries like OpenCV or PIL. Please include detailed comments explaining each step of the code.
```

AI Response*:
```
Here's a Python program that outlines the process of analyzing images for traits like surface artifacts, anatomical inaccuracies, and context clues using OpenCV and PIL (Pillow). The code includes comments to explain each step. Keep in mind that this example is a prototype and would require significant refinement and dataset training for practical use.

Ai Image Analysis

This program provides a starting point for analyzing images to differentiate between AI-generated and human-generated ones. Key features include:

Surface Artifacts Detection: Uses edge detection to highlight unusual transitions.
Anatomical Analysis: Finds contours to detect irregular shapes potentially present in AI-generated images.
Background Consistency: Applies k-means clustering to detect uniform background colors.
Make sure to test it with diverse image samples for meaningful results. Let me know if you'd like enhancements or further explanations.
```
*Code provided in "AI_Image_Analysis.py" file.

The initial code did not work. It was simply analyzing the different traits and then trying to display the analyzed image (which didn't work in VS code). After clarifying the requirement and asking it to rewrite the code, it seemed to run as expected, with varied results.

## Step 5: Run the Program and Test It
human_ex1
  - Response: The image is likely AI-generated.

human_ex2
  - Response: The image is likely AI-generated.

human_ex3
  - Response: The image is likely AI-generated.


AI_ex1
  - Response: The image is likely AI-generated.

AI_ex2
  - Response: The image is likely AI-generated.

AI_ex3
  - Response: The image is likely AI-generated.


The program incorrectly labeled all of the human generated images as AI, but correctly labeled all the AI generated images as AI. Either that, or reddit artists are secretely using AI for all their commissions (but I highly doubt that based on the quality of the artwork selected)!

## Step 6: Reflect and Analyze

### 1. Program Performance:
How well did the program differentiate between AI-generated and human-generated media?
  - The program did not differentiate between AI and human generated images very well.
Were there any examples it misclassified?
  - It misclassified all of the human generated images.

### 2. Feature Analysis:
What aspects of the media (e.g., symmetry, coherence, artifacts) did the program analyze?
1. Surface Artifacts
2. Anatomical Inaccuracies
3. Background and Context Clues
   
Why do you think these aspects were useful (or not useful) for detecting AI generation?
  - I don't think any of these aspects were that useful due to the images that I selected, because they were quite detailed, so they were likely too complex for the simple program generated.

### 3. Limitations and Improvements:
What limitations did you notice in the program’s approach?
  - In the code itself it says it is a "Basic heuristic to classify image as AI-generated or not (This is a naive thresholding approach for demonstration purposes)." I'm sure I could fine tune the scores that it is judging against, but it also doesn't have any data to go off besides the basic scores, which is a big limitation. This example may also be limited by the images that I provided. They all have a lot going on in them, so even though they are high quality, they could be too much to analyze with this simple approach.

How could the detection program be improved?
  - My own guess would be to try fine-tuning the scores until it labeled my images correctly, then test it out on other images. Alternatively, we could improve how it analyzes each of the traits based on more data.
  - In the code response from ChatGPT, it noted that this is a basic approach, and to let it know if I wanted help refining it to make it more accurate. When asked how I could do that, this was ChatGPT's response:
```
To refine the heuristic approach and improve accuracy, here are several suggestions for enhancements:

1. Collect and Label a Dataset
Gather a diverse dataset of both AI-generated and human-generated images.
Label the dataset correctly to use it for testing and tuning.
2. Feature Engineering
Analyze more detailed traits from the image:

Color Distributions: Analyze pixel intensity histograms for unusual patterns.
Texture Analysis: Use techniques like Local Binary Patterns (LBP) to capture fine-grained details.
Anomaly Detection: Compute differences between object structures (e.g., eyes, hands) compared to human anatomical norms.
3. Machine Learning Models
Replace simple heuristic thresholds with trained classifiers:

Support Vector Machines (SVM) or Random Forests: Train them using extracted features (surface artifacts, edge counts, contour counts, color variances).
Neural Networks: Fine-tune pre-trained image models like ResNet or MobileNet for binary classification.
4. Use Pre-Trained Image Models (Transfer Learning)
Use CNN-based models from TensorFlow or PyTorch to learn more complex image patterns.
Example approach:
python
Copy
Edit
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array

# Example for feature extraction
model = MobileNetV2(weights='imagenet', include_top=False)
features = model.predict(np.expand_dims(img_to_array(image), axis=0))
5. Advanced Image Quality Checks
Use these image checks as new features:

Surface Noise Patterns: Detect pixelation and digital noise typical of AI images.
Face and Object Detection: Use tools like dlib or mediapipe to detect odd facial features or extra limbs.
Background Symmetry: Identify perfect symmetries or repetitive structures, which may hint at AI generation.
6. Statistical Analysis and Threshold Refinement
Use cross-validation on your dataset to adjust thresholds for better classification accuracy. Examples:

Fine-tune contour count thresholds for complex objects.
Improve edge-detection limits by analyzing false positives and negatives.
Would you like code examples for integrating a machine learning model or tools like TensorFlow? I'd be happy to guide you further!
```

