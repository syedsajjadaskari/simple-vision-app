"""
Simple Azure Vision Streamlit App
Upload images and analyze them with AI!
"""

import streamlit as st
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Azure Vision AI",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Azure Vision AI Demo")
st.markdown("Upload an image and let AI analyze it!")

# Sidebar for credentials
st.sidebar.header("⚙️ Azure Settings")
endpoint = st.sidebar.text_input("Endpoint", value="", type="default")
api_key = st.sidebar.text_input("API Key", value="", type="password")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True, width="stretch")

with col2:
    st.header("🎯 AI Analysis")
    
    if uploaded_file and endpoint and api_key:
        # Analyze button
        if st.button("🚀 Analyze Image", type="primary"):
            
            with st.spinner("Analyzing..."):
                try:
                    # Create client
                    client = ImageAnalysisClient(
                        endpoint=endpoint,
                        credential=AzureKeyCredential(api_key)
                    )
                    
                    # Convert image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    image_data = img_byte_arr.getvalue()
                    
                    # Analyze
                    result = client.analyze(
                        image_data=image_data,
                        visual_features=[
                            VisualFeatures.CAPTION,
                            VisualFeatures.TAGS,
                            VisualFeatures.OBJECTS
                        ]
                    )
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Caption
                    if result.caption:
                        st.subheader("📸 AI Caption")
                        st.info(f"**{result.caption.text}**")
                        st.caption(f"Confidence: {result.caption.confidence:.1%}")
                    
                    # Tags
                    if result.tags:
                        st.subheader("🏷️ Tags")
                        tags_text = ", ".join([f"{tag.name} ({tag.confidence:.0%})" 
                                             for tag in result.tags.list[:8]])
                        st.write(tags_text)
                    
                    # Objects
                    if result.objects:
                        st.subheader("🎯 Objects Detected")
                        for obj in result.objects.list[:5]:
                            st.write(f"• {obj.tags[0].name} - {obj.tags[0].confidence:.0%}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    elif uploaded_file:
        st.warning("⚠️ Please enter your Azure credentials in the sidebar")
    else:
        st.info("👈 Upload an image to get started")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 How to Use")
st.sidebar.markdown("""
1. Enter your Azure Vision credentials
2. Upload an image
3. Click 'Analyze Image'
4. See AI results!
""")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit & Azure AI")
