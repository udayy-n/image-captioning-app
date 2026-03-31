import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Layer

# =========================
# CUSTOM ATTENTION
# =========================
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# =========================
# CONFIG
# =========================
MAX_LENGTH = 33

# =========================
# LOAD COMPONENTS
# =========================
@st.cache_resource
def load_components():

    model = load_model(
        "best_model_attn.keras",
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    base_model = InceptionV3(weights="imagenet")
    cnn_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("mixed10").output
    )

    return model, tokenizer, cnn_model

model, tokenizer, cnn_model = load_components()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(image):

    image = image.resize((299, 299))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = cnn_model.predict(image, verbose=0)

    features = features.reshape((features.shape[0], -1, features.shape[-1]))

    return features

# =========================
# INDEX → WORD
# =========================
def index_to_word(index, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    return None

# =========================
# GREEDY SEARCH
# =========================
def generate_caption(model, tokenizer, photo, max_length):

    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = index_to_word(yhat, tokenizer)
        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text

# =========================
# BEAM SEARCH
# =========================
def generate_caption_beam(model, tokenizer, photo, max_length, beam_width=3):

    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]

    for _ in range(max_length):

        all_candidates = []

        for seq, score in sequences:

            sequence = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([photo, sequence], verbose=0)[0]

            top_k = np.argsort(yhat)[-beam_width:]

            for word_idx in top_k:
                candidate = [
                    seq + [word_idx],
                    score - np.log(yhat[word_idx] + 1e-10)
                ]
                all_candidates.append(candidate)

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    best_seq = sequences[0][0]

    caption = []
    for idx in best_seq:
        word = index_to_word(idx, tokenizer)
        if word is None:
            continue
        if word == "endseq":
            break
        if word != "startseq":
            caption.append(word)

    return " ".join(caption)

# =========================
# CLEAN
# =========================
def clean_caption(text):
    text = text.replace("startseq", "")
    text = text.replace("endseq", "")
    return text.strip()

# =========================
# UI
# =========================
st.set_page_config(page_title="Image Caption Generator")

st.title("Image Caption Generator (Attention + Beam Search)")
st.write("Upload an image to generate captions")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Captions"):

        try:
            with st.spinner("Generating captions..."):

                features = extract_features(image)

                greedy_caption = clean_caption(
                    generate_caption(model, tokenizer, features, MAX_LENGTH)
                )

                beam_caption = generate_caption_beam(
                    model, tokenizer, features, MAX_LENGTH, beam_width=3
                )

                beam_caption = clean_caption(beam_caption)

            st.success("Captions Generated")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Greedy")
                st.write(greedy_caption)

            with col2:
                st.subheader("Beam Search")
                st.write(beam_caption)

        except Exception as e:
            st.error("Error during caption generation")
            st.write(str(e))