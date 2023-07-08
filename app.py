from flask import Flask, request, render_template, jsonify
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_model_output', methods=['POST'])
def get_model():
    data = request.get_json()
    text = data['text']
    
    model = TFAutoModelForSequenceClassification.from_pretrained('models/transformer/', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')


    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='tf',
        padding='max_length',
        truncation=True,
        max_length=331
    )

    outputs = model.predict(inputs)

    # Get the predicted label and confidence
    predicted_label = outputs.logits.argmax(axis=1).numpy()[0]
    confidence = tf.nn.softmax(outputs.logits, axis=1).numpy()[0][predicted_label]

    # Return the prediction as JSON
    response = {'predicted_label': predicted_label, 'confidence': confidence}
    return jsonify(response)

@app.route('/get_rnn_output', methods=['POST'])
def get_rnn_output():
    data = request.get_json()
    text = data['text']

    model = tf.keras.models.load_model('models/rnn.h5')
    output = model.predict(text)
    outputs = np.argmax(output, axis=1)

    return jsonify({'results': outputs})

if __name__ == '__main__':
    app.run(debug=True)