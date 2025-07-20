#🌿 Leaf Disease Detection
A deep learning project that employs a Convolutional Neural Network (CNN) to detect and classify diseases in plant leaves. With a web-friendly interface, this system enables users—especially in agriculture—to upload leaf images and get disease predictions, helping to improve crop health and yield.

📁 Repository Structure
text
Copy
Edit
Leaf-Disease-Detection/
├── Train_plant_disease.ipynb       # Notebook for training the CNN
├── Test_Plant_Disease.ipynb        # Notebook for testing/evaluation
├── main.py                         # Script for model inference/API/
├── training_hist.json              # JSON: model training history (loss/accuracy)
├── requirement.txt                 # Required Python packages
├── home_page.jpeg                  # Screenshot/sample UI image
└── README.md                       # Project documentation
📦 Dataset
Utilizes the New Plant Diseases Dataset from Kaggle:

~87,000 images covering 38 classes

Includes both healthy leaves and various diseases

📥 Download from:
Kaggle – New Plant Diseases Dataset

🧠 Model Architecture
Framework: TensorFlow + Keras

CNN structure: Conv2D → MaxPooling → Dropout → Dense

Loss: categorical_crossentropy

Optimizer: Adam

Output: Softmax over 38 class labels

The script main.py handles image input, preprocessing, and inference, making it suitable for integrating into web or mobile apps.

🚀 Quick Start
1. Clone the Repo & Install Requirements
bash
Copy
Edit
git clone https://github.com/jayantkumar1604/Leaf-Disease-Detection.git
cd Leaf-Disease-Detection
pip install -r requirement.txt
2. Prepare the Dataset
Download and extract the Kaggle dataset.

Place the unzipped folder in the project directory—ensure paths are updated in Jupyter notebooks.

3. Train the Model (Optional)
bash
Copy
Edit
jupyter notebook Train_plant_disease.ipynb
4. Evaluate/Test the Model
Use:

Test_Plant_Disease.ipynb for detailed evaluation.

main.py to run inference on single images or in API mode.

📊 Training Insights
Check training_hist.json for details on:

Training and validation loss

Accuracy trends over epochs
Visualize these trends in the Jupyter notebooks to monitor overfitting or convergence.

🖼️ UI Preview
Here's a snapshot of the UI/landing page using your sample image:



🌟 Next Steps
🔧 Build a front-end using Flask, Streamlit, or FastAPI

📲 Export model to TensorFlow Lite for mobile compatibility

📌 Add explainability via Grad-CAM or saliency maps

🎯 Augment data with GANs (e.g., LeafGAN) or apply transfer learning for higher accuracy 

⚙️ Integrate real-time camera input (mobile or web)

🙋‍♂️ Author
Jayant Kumar
jayantkumar1604@gmail.com
Data Set Link = https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
