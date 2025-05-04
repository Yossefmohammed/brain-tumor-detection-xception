# Brain Tumor Detection using Xception Model

## Overview
This project implements a deep learning model for brain tumor detection using the Xception architecture. The model is trained to classify brain MRI images into two categories: tumor present ('yes') or no tumor present ('no').

## Contact Information
- Email: ypsefmohammedahmed@gamil.com
- Phone: 01126078938

## Dataset
The project uses the Brain MRI Images for Brain Tumor Detection dataset from Kaggle, which contains:
- Images of brain MRIs
- Two classes: 'yes' (tumor present) and 'no' (no tumor present)
- Total of 253 images split into training, validation, and test sets

## Model Architecture
The project uses the Xception model, a deep convolutional neural network architecture that:
- Utilizes depthwise separable convolutions
- Has 71 layers
- Achieves high performance with relatively fewer parameters
- Is pre-trained on ImageNet and fine-tuned for brain tumor detection

## Implementation Details
1. **Data Preprocessing**:
   - Image resizing to 256x256 pixels
   - Data augmentation using ImageDataGenerator
   - Image enhancement using custom preprocessing function
   - Train/validation/test split (70%/15%/15%)

2. **Model Training**:
   - Uses transfer learning with Xception base model
   - Custom top layers added for binary classification
   - Adam optimizer with learning rate 0.001
   - Batch size of 64
   - Early stopping to prevent overfitting

3. **Image Enhancement**:
   - Custom preprocessing function to enhance image quality
   - Includes contrast adjustment
   - Sharpening using kernel convolution
   - Value channel enhancement

## Results
The model achieves high accuracy in detecting brain tumors from MRI images. Detailed performance metrics and visualizations are available in the notebook.

## Requirements
- Python 3.10.13
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage
1. Clone the repository
2. Install required dependencies
3. Download the dataset from Kaggle
4. Run the Jupyter notebook to train and evaluate the model

## License
This project is open source and available under the MIT License.

## Acknowledgments
- Dataset: Brain MRI Images for Brain Tumor Detection from Kaggle
- Xception model architecture by François Chollet
