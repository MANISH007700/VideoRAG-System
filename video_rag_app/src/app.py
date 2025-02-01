import logging
import os
import time
from pathlib import Path

import streamlit as st
import yaml
from inference import InferenceProcessor
from retriever import VideoRetriever
from utils.helpers import cleanup_data_directories
from utils.logger import setup_logger
from video_indexer import VideoIndexer
from video_processor import VideoProcessor

# Setup logger
logger = setup_logger()

# Custom CSS styling
STYLE = """
<style>
    .main {background-color: white;}
    h1 {color: darkslategray; border-bottom: 2px solid darkslategray;}
    .stButton>button {background-color: #2196F3; color: white; border-radius: 5px;}
    .stTextInput>div>div>input {border: 1px solid darkslategray; background-color: white;}
    .stProgress>div>div>div {background-color: #4CAF50;}
    .sidebar .sidebar-content {background-color: white;}
    .log-box {padding: 10px; margin: 10px 0; border-radius: 5px; background-color: white; border: 1px solid #e0e0e0;}
    .api-key-popup {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);}
</style>
"""


def load_config():
    """
    Load and parse the application configuration from the YAML file.

    Returns:
        dict: Configuration dictionary containing all settings from config.yaml

    Raises:
        FileNotFoundError: If config.yaml is not found in the config directory
        yaml.YAMLError: If the YAML file is malformed or cannot be parsed
    """
    config_path = "config/config.yaml"
    logger.info(f"Loading configuration from {config_path}")

    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.debug(f"Successfully loaded configuration: {config}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise

def init_session_state():
    """
    Initialize the Streamlit session state with required keys and default values.
    This ensures all necessary variables are available throughout the app's lifecycle.
    
    The following keys are initialized if not already present:
    - video_url: Stores the URL of the uploaded/processed video
    - index: Stores the vector index created from video frames and captions
    - retriever: Stores the VideoRetriever instance for querying the index
    - video_id: Stores unique identifier for the processed video
    - inference_processor: Stores the Gemini model interface
    - gemini_key: Stores the user's Gemini API key
    """
    logger.debug("Initializing session state variables")
    
    # Define required session state keys and their default values
    required_keys = {
        "video_url": None,  # URL of the video being processed
        "index": None,      # Vector index for search
        "retriever": None,  # Video retrieval interface
        "video_id": None,   # Unique video identifier
        "inference_processor": None,  # Gemini model interface
        "gemini_key": None, # API key for Gemini
    }

    # Initialize any missing keys in session state
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            logger.debug(f"Initializing session state key: {key}")
            st.session_state[key] = default_value


def main():

    st = time.time()

    logger.info("Starting Video RAG System application + Builtin Voice Mode Activated")
    st.set_page_config(page_title="Video RAG System", layout="wide", page_icon="🎥")
    st.markdown(STYLE, unsafe_allow_html=True)

    # Initialize session state
    logger.debug("Initializing session state")
    init_session_state()

    # API Key Popup
    if not st.session_state.gemini_key:
        logger.info("No Gemini API key found - displaying key input form")
        with st.container():
            st.markdown("<div class='api-key-popup'>", unsafe_allow_html=True)
            st.header("🔑 Gemini API Key Required")
            api_key = st.text_input(
                "Please enter your Gemini API key to continue:", type="password"
            )
            cols = st.columns([1, 3, 1])
            with cols[1]:
                if st.button("Submit Key"):
                    if api_key:
                        logger.info("API key submitted successfully")
                        st.session_state.gemini_key = api_key
                        st.rerun()
                    else:
                        logger.warning("Empty API key submitted")
                        st.error("Please enter a valid API key")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()  # Stop execution until key is entered

    # Load configuration
    logger.debug("Loading configuration from YAML")
    config = load_config()

    # Initialize InferenceProcessor
    if st.session_state.inference_processor is None:
        try:
            logger.info("Initializing InferenceProcessor")
            st.session_state.inference_processor = InferenceProcessor(
                st.session_state.gemini_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize InferenceProcessor: {str(e)}")
            st.error(f"❌ Failed to initialize API: {str(e)}")
            del st.session_state.gemini_key
            st.rerun()

    # Cleanup Section
    st.sidebar.header("Settings ⚙️")
    if st.sidebar.button("🧹 Cleanup All Data"):
        try:
            logger.info("Starting cleanup of data directories")
            with st.spinner("Cleaning up previous data..."):
                cleanup_data_directories()
            # Reset only processing-related states
            reset_keys = ["video_url", "index", "retriever", "video_id"]
            for key in reset_keys:
                st.session_state[key] = None
            logger.info("Cleanup completed successfully")
            st.success("All previous data cleaned successfully!")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
            st.error(f"Error during cleanup: {str(e)}")

    # Main Content
    st.title("🎥 Video RAG System")

    # Video Processing Section
    with st.container():
        st.header("Step 1: Process YouTube Video 🎬")
        video_url = st.text_input("Enter YouTube URL:")
        process_button = st.button("🚀 Process Video")

    if process_button and video_url:
        logger.info(f"Starting video processing for URL: {video_url}")
        try:
            status_container = st.container()
            progress_bar = st.progress(0)

            with status_container:
                st.markdown("### Processing Steps 📋")
                log_box = st.empty()

            def update_log(message):
                logger.debug(f"Processing status: {message}")
                log_box.markdown(
                    f'<div class="log-box">📌 {message}</div>', unsafe_allow_html=True
                )

            with st.spinner("Processing video..."):
                update_log("Initializing video processor...")
                video_processor = VideoProcessor(video_url, config)
                progress_bar.progress(5)

                # Add progress callback
                def handle_progress(status):
                    update_log(status)

                update_log("Starting video download...")
                try:
                    metadata, video_path = video_processor.download_video(
                        progress_callback=handle_progress
                    )
                except Exception as e:
                    logger.error(f"Video download failed: {str(e)}", exc_info=True)
                    st.error(f"❌ Download failed: {str(e)}")
                    return

                progress_bar.progress(25)

                # Convert duration to minutes:seconds format
                def format_duration(seconds):
                    return f"{seconds//60}:{seconds%60:02d}"

                video_duration = format_duration(metadata.duration)
                update_log(
                    f"Download complete: {video_path.name} - Duration: {video_duration}"
                )

                update_log("Extracting frames from video...")
                frames_dir = video_processor.extract_frames(video_path)
                progress_bar.progress(50)
                update_log(f"Extracted {len(list(frames_dir.glob('*.jpg')))} frames")

                update_log("Extracting video captions...")
                captions_path = video_processor.extract_captions()
                progress_bar.progress(70)
                update_log(f"Captions saved to: {captions_path}")

                update_log("Creating multimodal index...")
                indexer = VideoIndexer(config)
                index = indexer.create_multimodal_index(
                    frames_dir, captions_path, video_processor.video_id
                )
                progress_bar.progress(90)
                update_log("Index creation complete")

                st.session_state.index = index
                st.session_state.video_url = video_url
                st.session_state.video_id = video_processor.video_id
                st.session_state.retriever = VideoRetriever(index)

                progress_bar.progress(100)
                logger.info("Video processing completed successfully")
                st.success("✅ Video processed successfully!")

        except Exception as e:
            error_msg = f"❌ Error processing video: {str(e)}"
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            st.error(error_msg)
            return

    # Query Section
    if st.session_state.index is not None:
        st.header("Step 2: Query Video Content 🔍")
        if st.session_state.video_url:
            st.video(st.session_state.video_url)

        query = st.text_input("Enter your query:")
        if st.button("📤 Submit Query"):
            logger.info(f"Processing query: {query}")
            try:
                processing_container = st.container()
                with processing_container:
                    st.markdown("### Query Processing Steps 📋")
                    query_log_box = st.empty()
                    query_progress = st.progress(0)

                def update_query_log(message):
                    logger.debug(f"Query processing status: {message}")
                    query_log_box.markdown(
                        f'<div class="log-box">📌 {message}</div>',
                        unsafe_allow_html=True,
                    )

                with st.spinner("Analyzing query..."):
                    update_query_log("Starting query processing...")
                    query_progress.progress(20)

                    update_query_log("Searching for relevant content...")
                    retrieved_images, retrieved_texts = (
                        st.session_state.retriever.retrieve(query)
                    )
                    query_progress.progress(40)
                    update_query_log(
                        f"Found {len(retrieved_images)} relevant frames and {len(retrieved_texts)} text segments"
                    )

                    update_query_log("Generating response with Gemini...")
                    response = st.session_state.inference_processor.process_query(
                        retrieved_images, retrieved_texts, query
                    )
                    query_progress.progress(80)

                    st.subheader("Answer 💡")
                    st.markdown(f"**{response['answer']}**")

                    st.subheader("Retrieved Frames 🖼️")
                    num_cols = min(3, len(retrieved_images))
                    cols = st.columns(num_cols)
                    for idx, image_path in enumerate(retrieved_images):
                        with cols[idx % num_cols]:
                            st.image(str(image_path), use_container_width=True)
                            st.caption(f"Frame {idx + 1}")

                    query_progress.progress(100)
                    logger.info("Query processing completed successfully")
                    update_query_log("Query processing complete!")

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                st.error(f"❌ Error processing query: {str(e)}")

    et = time.time()
    print("Time taken - ", round(et - st, 3))


if __name__ == "__main__":
    main()
