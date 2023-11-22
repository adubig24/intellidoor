from flask import Flask, request
from feature_extraction import feature_extraction
from werkzeug.utils import secure_filename
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/api/prediction", methods=['GET', 'POST'])
def get_prediction_result():
    if request.method == 'GET':
        return "Connexion OK", 200
    elif request.method == 'POST':
        if 'image.jpg' not in request.files:
            return 'No image file in request', 400

        file = request.files['image.jpg']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        extractor = feature_extraction()
        feature = extractor.feature_extraction(img)
        dataframe = extractor.transformation_dataframe(feature)
        extractor.sauvegarde_csv(dataframe, img)

        return "OK", 200
@app.errorhandler(404)
def page_not_found(e):
    return "erreur", 404

if __name__ == "__main__" :
    app.run(debug=False, port=5001)