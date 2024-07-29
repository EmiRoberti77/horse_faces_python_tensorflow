from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('horse_face_markings_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_path = f"./{img_file.filename}"
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    traits = ['Blaze', 'Stripe', 'Bald face', 'Star', 'Snip']
    predicted_trait = traits[np.argmax(prediction)]

    return jsonify({'trait': predicted_trait})

if __name__ == '__main__':
    app.run(debug=True)
