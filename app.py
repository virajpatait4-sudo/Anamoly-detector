import streamlit as st
import pandas as pd
import joblib
from collections import Counter


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

detector = joblib.load("syscall_anomaly_detector.pkl")

model = detector["model"]
threshold = detector["threshold"]
feature_columns = detector["feature_columns"]


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(sequence):

    features = {}

    # Sequence length
    features["sequence_length"] = len(sequence)

    # Syscall frequency
    syscall_counts = Counter(sequence)

    for syscall, count in syscall_counts.items():
        features[f"syscall_{syscall}"] = count

    # Bigram features
    for i in range(len(sequence) - 1):

        bigram = (sequence[i], sequence[i + 1])

        key = f"bigram_{bigram[0]}_{bigram[1]}"

        features[key] = features.get(key, 0) + 1

    # Trigram features
    for i in range(len(sequence) - 2):

        trigram = (
            sequence[i],
            sequence[i + 1],
            sequence[i + 2]
        )

        key = f"trigram_{trigram[0]}_{trigram[1]}_{trigram[2]}"

        features[key] = features.get(key, 0) + 1

    return features


# --------------------------------------------------
# STREAMLIT INTERFACE
# --------------------------------------------------

st.title("ML-Based System-Call Anomaly Detector")

st.write(
    "Enter a Linux system-call sequence to determine "
    "whether the behaviour is normal or anomalous."
)


# Input box

syscall_input = st.text_area(
    "Enter system-call sequence",
    placeholder="Example: 5 3 3 4 5 197 192 3 4"
)


# Predict button

if st.button("Detect Anomaly"):

    if syscall_input.strip() == "":
        st.warning("Please enter a system-call sequence.")

    else:

        try:

            # Convert input into integers
            sequence = [
                int(x)
                for x in syscall_input.split()
            ]

            # Extract features
            features = extract_features(sequence)

            # Convert to DataFrame
            input_df = pd.DataFrame([features])

            # Make columns identical to training data
            input_df = input_df.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # Calculate anomaly score
            score = model.decision_function(input_df)[0]

            # Prediction
            if score < threshold:
                prediction = "ANOMALY"
            else:
                prediction = "NORMAL"


            # --------------------------------------------------
            # DISPLAY RESULT
            # --------------------------------------------------

            st.subheader("Detection Result")

            st.metric(
                "Anomaly Score",
                round(score, 4)
            )

            if prediction == "ANOMALY":

                st.error("⚠️ ANOMALY DETECTED")

                st.write(
                    "The syscall sequence differs from the "
                    "normal behaviour learned by the model."
                )

            else:

                st.success("✅ NORMAL BEHAVIOUR")

                st.write(
                    "The syscall sequence is consistent with "
                    "the learned normal behaviour."
                )


        except ValueError:

            st.error(
                "Invalid input. Please enter only syscall numbers "
                "separated by spaces."
            )