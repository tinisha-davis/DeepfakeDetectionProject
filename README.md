# Team-TAPM
Team TAPM the for Hack4Impact Hackathon April 26th, 2025
Our project is for a Deepfake Detection extension for any browser.

### Team members:

-Tinisha\
-Alan\
-Psy\
-Mycole

### Project Directory
```
deepfake_detector/
├── data/
│ ├── real/
│ ├── fake/
│ └── test/
│ └── sample_image.jpg
├── models/
│ └── deepfake_detector.h5
├── output/
└── p.py
```

### Other ideas

### Set up your environment:
Install Python and required libraries (TensorFlow, OpenCV, NumPy, Matplotlib, scikit-learn)\
Create a virtual environment for your project\

### Obtain a dataset:
You'll need labeled data of real and fake images/videos\
Popular datasets include FaceForensics++, Deepfake Detection Challenge dataset, or DFDC\

### Prepare your directory structure:

Create folders for real and fake images

Set up directories for training, validation, and test data\

### Modify the paths in the code to point to your dataset locations:
```
python
real_dir = "your/path/to/real/images"
fake_dir = "your/path/to/fake/images"
```

### Run the training process:
Execute the main function to train your model
This will take time depending on your hardware and dataset size
Test your model with new images or videos not used in training
### Iterate and improve:

Experiment with different model architectures
Try different hyperparameters
Improve preprocessing techniques

## Project Summary
The deepfake detection project works as follows:
Data Preparation: The system extracts faces from videos or processes images, creating a dataset of real and manipulated faces.
Transfer Learning: The model uses MobileNetV2 (pre-trained on ImageNet) as a feature extractor, then adds custom classification layers on top.

## Training Process:
The neural network learns to distinguish between authentic and synthetic media through supervised learning with labeled examples.

## Detection Method: 
The model analyzes visual artifacts and inconsistencies that deepfake algorithms typically leave behind, such as unnatural blending, lighting inconsistencies, or unusual facial movements.

## Classification: 
For each input image, the model outputs a probability score between 0 and 1, where scores closer to 1 indicate a real image and scores closer to 0 suggest a fake.

The project could be extended with a user interface, batch processing capabilities, or integration with video platforms for real-time detection.
