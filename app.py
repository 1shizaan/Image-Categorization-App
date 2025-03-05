# app.py
import streamlit as st
import os
from PIL import Image
from categorizer import ImageCategorizer  # or wherever you placed the class

def main():
    st.title("Image Categorization App")

    # Instantiate the categorizer
    categorizer = ImageCategorizer()

    st.write("Upload one or more images to categorize them.")

    # Let users upload multiple images
    uploaded_files = st.file_uploader("Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Create a list of file paths to pass to our categorizer
        file_paths = []

        # Save each uploaded file to a temp folder so we can pass a path
        for uploaded_file in uploaded_files:
            # Convert the uploaded file to an actual file on disk
            temp_path = os.path.join("temp_uploads", uploaded_file.name)
            os.makedirs("temp_uploads", exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_paths.append(temp_path)

        # Process/categorize images
        results = categorizer.process_images(file_paths)

        # Show the results
        for path, info in results.items():
            st.subheader(f"File: {os.path.basename(path)}")
            # Display the image
            st.image(path, use_column_width=True)

            if info["domain"] != "Error":
                st.write(f"**Predicted Domain:** {info['domain']}")
                st.write(f"**Probability:** {info['probability']:.2f}")
                st.write(f"**Top Domains:** {', '.join(info['top_domains'])}")
                if info["saved"]:
                    st.write(f"**Saved to:** `{info['saved_path']}`")
                else:
                    st.write("Not saved (probability too low or error).")
            else:
                st.write("Error categorizing this image.")

if __name__ == "__main__":
    main()
