from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
import librosa

app = Flask(__name__)
model = tf.keras.models.load_model(r'C:\Users\HP\OneDrive\Desktop\LABS\DSP\model.h5')

label_dict = {0: 'Pig', 1:'Bat', 2:'Elephant', 3:'Rat' }


def melspectrogram(audio_file, n_mels=128, hop_length=512, n_fft=2048):
    y, sr = librosa.load(audio_file , duration=30.0)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, window='hann')
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Min-max scaling
    mel_spectrogram -= np.min(mel_spectrogram)
    mel_spectrogram /= np.max(mel_spectrogram)

    return mel_spectrogram


def preprocess_testing(file, n_mels=128, hop_length=512, n_fft=2048):
    X = []
    if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.ogg'):
        mel_spec = melspectrogram(file, n_mels=n_mels, hop_length=hop_length)

        if mel_spec.shape[1] >= 1292:  
            mel_spec = mel_spec[:, :1292]
            X.append(mel_spec)
        else:
            pad_width = ((0, 0), (0, 1292 - mel_spec.shape[1]))
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
            X.append(mel_spec)

    X = np.array(X)
    X = X[..., np.newaxis]
    return X


@app.route('/')
def html_file():
    return render_template('Main.html')


@app.route('/process_audio', methods=['POST'])
def processAudio():

    image_upload = request.files['image']
    file_path = 'Imagefile/' + image_upload.filename

    # Create a directory if it doesn't exist
    if not os.path.exists('Imagefile/'):
        os.makedirs('Imagefile/')

    image_upload.save(file_path)

    # Load the image and preprocess it
    y = preprocess_testing(file_path)

    # Make predictions using the Wildfire Model
    predictions = label_dict[np.argmax(model.predict(y, batch_size=1))] 

    os.remove(file_path)
    print(predictions)
    # Process the predictions
    result = predictions

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
