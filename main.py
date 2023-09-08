from flask import Flask
import torchvision
import tensorflow as tf

app = Flask(__name__)


class SegmentationModel():
    def __init__(self, model_path = "App/ml_models/segmodelv3.tflite"):
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
        input_data = torchvision.transforms.Resize((512,512))(input_data)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        # get_tensor() returns a copy of the tensor data
        # use tensor() in order to get a pointer to the tensor
        leaf_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        disease_data = self.interpreter.get_tensor(self.output_details[1]['index'])
        leaf_data = torch.from_numpy(leaf_data)
        disease_data = torch.from_numpy(disease_data)

        return (leaf_data, disease_data)


@app.route('/')
def index():
    return 'Hello from Flask!'

if __name__ == 'main':
  app.run(host='0.0.0.0', port=81)
