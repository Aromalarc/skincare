import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import tflite_runtime.interpreter as tflite

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load TFLite model
interpreter = tflite.Interpreter(model_path="static/model/mobilenet_skin_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 128
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

explanation_map = {
    'akiec': "Actinic keratoses are precancerous skin lesions caused by sun damage.",
    'bcc': "Basal cell carcinoma is a common skin cancer that grows slowly and rarely spreads.",
    'bkl': "Benign keratosis-like lesions are non-cancerous skin growths.",
    'df': "Dermatofibroma is a benign skin nodule, often caused by minor injury.",
    'mel': "Melanoma is a serious form of skin cancer that requires urgent medical attention.",
    'nv': "Melanocytic nevi are common moles, generally harmless unless changing.",
    'vasc': "Vascular lesions are blood vessel growths, usually benign."
}

cream_map = {
    'mel': "Consult a dermatologist immediately. Do not self-treat.",
    'nv': "No treatment needed for common moles unless changes occur.",
    'bkl': "Hydrocortisone cream or cryotherapy may be used.",
    'bcc': "Consult for surgical or topical options like imiquimod.",
    'akiec': "May use fluorouracil or diclofenac-based creams.",
    'df': "Usually does not require treatment unless bothersome.",
    'vasc': "May be treated with laser therapy if needed."
}

def preprocess_image(image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return "No file uploaded."

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = preprocess_image(file_path)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        confidence = np.max(output_data)

        if confidence < 0.8:
            prediction = "Image does not appear to be a skin condition. Please try again with a clearer image."
            explanation = None
            cream = None
        else:
            pred_class = labels[np.argmax(output_data)]
            prediction = f"{pred_class} ({confidence:.2%} confidence)"
            explanation = explanation_map.get(pred_class, "No explanation available.")
            cream = cream_map.get(pred_class, "Consult a dermatologist for suitable treatment.")

        return render_template('result.html',
                               prediction=prediction,
                               explanation=explanation,
                               cream=cream,
                               file_path=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
