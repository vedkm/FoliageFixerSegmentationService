from flask import Flask, jsonify, request
import tensorflow as tf
import PIL
import io
import os

app = Flask(__name__)


class SegmentationModel():
    def __init__(self, model_path = "segmodelv3.tflite"):
        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def forward(self, x):
        input_data = x
        # input_data = torchvision.transforms.functional.convert_image_dtype(input_data, dtype=torch.float32)
        # input_data = torchvision.transforms.Resize((512,512))(input_data)
        input_data = tf.image.resize(input_data, [512,512])
        input_data = tf.transpose(input_data, perm=[2,0,1])
        input_data = input_data[tf.newaxis, ...]
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        # get_tensor() returns a copy of the tensor data
        # use tensor() in order to get a pointer to the tensor
        leaf_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        disease_data = self.interpreter.get_tensor(self.output_details[1]['index'])
        # leaf_data = torch.from_numpy(leaf_data)
        # disease_data = torch.from_numpy(disease_data)
        return (leaf_data, disease_data)

def FileStorage_to_Tensor(file_storage_object):
    # image_binary = file_storage_object.read()
    image_binary = file_storage_object
    pil_image = PIL.Image.open(io.BytesIO(image_binary))
    tensor_image = tf.convert_to_tensor(pil_image)
    return tensor_image

segmentation_model = SegmentationModel()

@app.route('/', methods=['POST'])
def index():
    try:
        data = request.form

        # Get the image file from the request
        image_file = request.files['image']

        image_data = image_file.read()

        '''
        Classification with models stored locally
        ** do not delete
        '''
        # Run the image through the classification model to get the prediction
        # step 1 load image as tensor
        image = FileStorage_to_Tensor(image_data)
        print(type(image))
        # step 2 segment image
        leaf, disease = segmentation_model.forward(image)

        return '{' + f'\"leaf\": \"{leaf.tostring()}\", \"disease\": \"{disease.tostring()}\"' + '}'
    except Exception as E:
        print(E)
        return jsonify({
            'error': E.__str__()
        }), 500


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=81)
    app.run(host='0.0.0.0')
