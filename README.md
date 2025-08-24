# STOP Sign Detection with OpenCV

This project detects STOP traffic signs using simple image processing:
- HSV color thresholding for red
- Morphological filtering
- Contour + shape scoring
- Bounding box + center point output

---

## Requirements
- Python 3.9+
- OpenCV (`pip install opencv-python`)
- NumPy

## How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/stop-sign-detection.git
   cd stop-sign-detection

2. python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

3. run this in terminal:
 python detect_stop.py --data stop_sign_dataset --out results --save-mask

4. python evaluate.py --pred results/detections.csv



