# Deepfake Detection System

This project is an AI-based system that detects whether a video is real or fake (deepfake).

## Features
- Detects deepfake videos using deep learning
- Supports multi-face detection
- Provides confidence score for predictions

## Technologies Used
- Python
- PyTorch
- OpenCV
- Transformers

## How to Run
1. source venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt
3. Run the project:
   for frontend--- python app.py
   for backend--
   python predict.py \
   --video_path test_final.mp4 \
   --config config/size_invariant_timesformer.yaml \
   --model_weights outputs/models/Model_checkpoint16 \
   --workers 0

   
