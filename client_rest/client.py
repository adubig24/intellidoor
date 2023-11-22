from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
from threading import Thread
from mtcnn import MTCNN
import requests
import time

global capture, switch, camera 
capture=0
switch=1

app = Flask(__name__, template_folder='./templates')

def call_api(image):
    URL_API="http://127.0.0.1:5001/api/prediction"
    imencoded = cv2.imencode(".jpg", image)[1]
    reponse = requests.post(URL_API, files={'image.jpg': imencoded.tostring()})
    return reponse

def extract_face(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    faces = []
    
    for res in results:
        x, y, width, height = res['box']
        x, y = abs(x), abs(y)
        diff = abs(width - height)
        if height > width:
            width = height
            x = x - diff//2
        else:
            height = width
            y = y - diff//2
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
        faces.append(face)

    return faces[0] if faces else None

camera = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                capture=0
                face = extract_face(frame)
                reponse = call_api(face)
                # with open("reponse.txt", "w") as f:
                #     f.write(reponse)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if switch==1:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        pass

@app.route('/requests', methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('stop') == 'Stop/Start':
            if switch==1:
                switch=0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch=1  
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.form.get('click') == 'Capture':
            global capture
            capture=1
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
    
camera.release()
cv2.destroyAllWindows()