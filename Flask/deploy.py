import base64
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

letras = np.array(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","Nothing","O","P","Q","R","S","Space","T","U","V","W","X","Y","Z"])
modelo = tf.keras.models.load_model('../Modelos/ModeloTrain.keras')

app = Flask(__name__, static_folder="./Templates/Static", template_folder='./Templates')
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins='*')

def base64_to_image(base64_string):
    
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})


@socketio.on("image")
def receive_image(image):

    # Decode the base64-encoded image data
    image = base64_to_image(image)

    frame_resized = cv2.resize(image, (70, 70))

    p = modelo.predict(np.array([frame_resized]))
    
    print(f"La clase predicha es {letras[np.argmax(p)]} con una probabilidad de {np.max(p)*100}%")

    cv2.putText(image, f"FPS: {letras[np.argmax(p)]}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
    # Display the image

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    frame_encoded = cv2.imencode(".jpg", image, encode_param)

    processed_img_data = base64.b64encode(frame_encoded).decode()

    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data

    emit("processed_image", processed_img_data)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":  
    socketio.run(app, debug=True, port=8000, host='0.0.0.0')