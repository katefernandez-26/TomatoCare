# TomatoCare

**A CNN-Based Web Application for Tomato Plant and Leaf Disease Detection with Treatment Recommendations Using Live Camera Capture and Image Upload**

TomatoCare is a Flask web application with a TensorFlow/Keras CNN image classifier. It supports leaf image uploads, live camera capture in supported browsers, predictions for classes included in the trained model, and general management/prevention recommendations.

## Requirements

- Python 3.10 or 3.11 is a practical starting point; TensorFlow support depends on OS and Python version.
- pip
- A labeled tomato leaf image dataset
- A laptop/desktop for easiest local development; Google Colab can be used to train the model.

## Setup on Windows

1. Install a compatible Python version.
2. Open this folder in VS Code.
3. Open the terminal in the project directory.
4. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Put labeled images into `dataset/`, with one folder per class. Example:

   ```text
   dataset/
     Tomato___Early_blight/
     Tomato___Late_blight/
     Tomato___healthy/
   ```

   Use a trusted, correctly labeled dataset. Include enough examples per class and keep a separate test set if evaluating performance formally.

7. Train the CNN:

   ```powershell
   python train_model.py
   ```

   This generates `tomato_disease_model.keras` and `class_names.json`.

8. Start the website:

   ```powershell
   python app.py
   ```

9. Open `http://127.0.0.1:5000` in your browser.

## Camera notes

Camera access usually requires HTTPS or localhost. Allow camera permission in the browser. If using a phone to access a server on another device, the camera may not work over a plain HTTP local-network address; use a secure HTTPS deployment or upload an image.

## Important limitations

- The model is not included because it must be trained on a suitable dataset.
- The starter CNN is for education/prototyping and is not a validated agricultural diagnostic tool.
- The model can only predict classes present in its training dataset. A high confidence score does not guarantee a correct diagnosis.
- Treatment tips are general guidance, not a substitute for local agricultural advice. Always follow product labels and local regulations.
- Do not deploy Flask's debug server publicly. Use a production WSGI server and secure hosting for deployment.
- Do not commit large datasets or trained model files to a public repository unless you have permission and have checked repository size limits.

## Project files

- `app.py`: Flask routes and prediction API
- `train_model.py`: CNN training script
- `templates/index.html`: web interface
- `static/style.css`: responsive styling
- `static/script.js`: upload, camera, and prediction interactions
- `requirements.txt`: Python dependencies
