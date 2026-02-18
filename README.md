SAPPHIRE-Road Segmentation System
Falcon is a deep learning project designed for autonomous navigation in off-road environments. It utilizes a DeepLabV3+ architecture with a ResNet50 backbone to perform semantic segmentation on 11 distinct natural classes. The system includes a Flask-based web interface for real-time terrain analysis.
Technical Stack
Deep Learning: PyTorch, Segmentation Models PyTorch (SMP)
Backbone: ResNet50 (ImageNet pre-trained)
Image Processing: OpenCV, Albumentations
Backend: Flask (Python)
Frontend: HTML5, CSS3
Segmentation Classes
The model identifies the following 11 categories to assist in pathfinding:
Sky, Trees, Lush Bushes, Dry Grass, Dry Bushes, Clutter, Flowers, Logs, Rocks, Landscape, and Background.
Project Structure
Plaintext
PROJECT5/
├── app.py                 # Flask server and model inference logic
├── dataset.py             # Custom PyTorch Dataset and class mapping
├── train.py               # Training script with Dice and Focal loss
├── evaluate.py            # mIoU metric calculation script
├── falcon_donkey_model.pth # Trained model weights
├── static/
│   └── style.css          # Frontend styling
└── templates/
    └── index.html         # Main web interface


Installation and Setup
1. Environment Setup
Clone the repository and install the required Python packages:
Bash
pip install flask torch torchvision segmentation-models-pytorch albumentations opencv-python tqdm

2. Model Training
To train the model using the Robust Loss (Dice + Focal + Weighted Cross-Entropy):
Bash
python train.py

3. Running the Web Interface
Start the Flask development server:
Bash
python app.py

Once the server is running, navigate to http://127.0.0.1:5000 in your web browser.

Implementation Details
Robust Loss Function
To handle the high class imbalance common in off-road datasets, the system uses a combined loss function:
Loss = DiceLoss + FocalLoss + WeightedCrossEntropy
This ensures the model focuses on rare classes like flowers and logs while maintaining accurate boundaries for trees and rocks.
Image Preprocessing
Input images are resized to 544x960 to ensure compatibility with the encoder's downsampling factor of 16. Normalization is applied using ImageNet mean and standard deviation values to ensure consistency with the pre-trained backbone.

Future Roadmap
Integration of NVIDIA TensorRT for real-time edge deployment.
Path-planning algorithm integration based on segmented "traversable" regions.
Support for video stream inference with temporal smoothing.

