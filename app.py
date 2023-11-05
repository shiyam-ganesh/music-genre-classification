import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import math
import io

@st.cache_resource
def load_model():
    mod = tf.keras.models.load_model('CNN_GTZAN.h5')
    return mod

model = load_model()

st.title("Music Genre Classification! 😃")

uploaded_audio = st.file_uploader("Upload a music you like 👇", type=["wav", "mp3"])

if uploaded_audio is not None:
    # Display the uploaded audio file
    st.audio(uploaded_audio, format='audio/wav', start_time=0)  # You can specify 'audio/mp3' for .mp3 files

    audio_data = uploaded_audio.read()

    SAMPLE_RATE = 22050
    TRACK_DURATION = 30 # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


    def preprocess(num_mfcc=40, n_fft=2048, hop_length=512, num_segments=10):

        # dictionary to store mapping, labels, and MFCCs
        data = {
        "mfcc": []
        }

        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

        # load audio file

        signal, sample_rate = librosa.load(io.BytesIO(audio_data), sr=SAMPLE_RATE)


        # process all segments of audio file
        for d in range(num_segments):

            # calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            # extract mfcc
            #mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

            mfcc = mfcc.T

            # store only mfcc feature with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
        return data
    
    Data = preprocess()
    x=np.array(Data["mfcc"])
    x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    #input_shape=x.shape[1:]
    Predictions = []

    def predict(model, X):
        """Predict a single sample using the trained model.
        :param model: Trained classifier
        :param X: Input data
        :param y: One-hot encoded target label
        """

        # add a dimension to input data for sample - model.predict() expects a 4d array in this case
        X = X[np.newaxis, ...]  # array shape (1, 130, 40, 1)

        # perform prediction
        prediction = model.predict(X)

        return prediction
    
    for segment in x:
        pred = predict(model, segment)
        Predictions.append(pred)
        #st.write(pred.shape)
        #st.write(pred)

    combined_array = np.vstack(Predictions)

    # Calculate the mean probabilities for each column
    mean_probabilities = np.mean(combined_array, axis=0)

    # Create column labels for the x-axis
    column_labels = ["Jazz", "Pop", "Rock", "Metal", "Blues", "Reggae", "Country", "Disco", "Classical", "Hiphop"]

    # Create the plot figure and axis explicitly
    fig, ax = plt.subplots(figsize=(10, 6))

    # Customize your plot using the 'ax' object
    ax.bar(column_labels, mean_probabilities, color='skyblue')
    ax.set_xlabel("Genres", fontsize=14)
    ax.set_ylabel("Mean Probabilities", fontsize=14)
    ax.set_title("Probability", fontsize=16, fontweight='bold')

    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)







