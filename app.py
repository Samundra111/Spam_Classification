import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model2.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Title
st.title("📧 Email Spam Classifier")

# Input
user_input = st.text_area("Enter the email content here:")
# In app.py, within the "Predict" block:
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Vectorize input (sparse)
        input_vector_sparse = vectorizer.transform([user_input])
        # Convert to dense array
        input_vector_dense = input_vector_sparse.toarray()

        # Predict using the dense input
        prediction = model.predict(input_vector_dense)[0]

        # Output
        if prediction == 1:
            st.error("🚨 This email is classified as SPAM.")
        else:
            st.success("✅ This email is NOT spam.")