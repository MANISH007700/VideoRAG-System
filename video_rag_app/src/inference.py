import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
from PIL import Image


class InferenceProcessor:
    def __init__(self, api_key: str):
        self.logger = logging.getLogger(__name__)
        genai.configure(api_key=api_key)

        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    # def _extract_timestamps(self, text: str) -> List[str]:
    #     """Extract timestamps from the caption text."""
    #     timestamp_pattern = r"<s>\s*([\d.]+)\s*:"
    #     timestamps = re.findall(timestamp_pattern, text)
    #     return sorted(list(set(timestamps)), key=float)  # Remove duplicates and sort

    def _extract_timestamps(self, text: str) -> List[str]:
        """Extract timestamps from the caption text using improved regex pattern matching.

        This method processes caption text that contains timestamps in the format:
        <s>START_TIME: caption text :END_TIME<e>

        Args:
            text (str): The caption text containing timestamp markers

        Returns:
            List[str]: A list of timestamp strings, where each timestamp is the
                      midpoint between the start and end times of a caption segment.
                      The timestamps are returned as strings to preserve decimal precision.

        Example:
            Input text: "<s>10.5: Some caption text :12.5<e>"
            Returns: ["11.5"] (average of 10.5 and 12.5)
        """
        # Match both start and end timestamps from the caption format
        # Pattern explanation:
        # <s>\s* - Match <s> followed by optional whitespace
        # ([\d.]+) - Capture group for decimal numbers (start time)
        # \s*: - Match colon with optional surrounding whitespace
        # .+? - Non-greedy match of caption text
        # : - Match ending colon
        # \s*([\d.]+)\s* - Capture group for decimal numbers (end time)
        # <e> - Match closing tag
        timestamp_pattern = r"<s>\s*([\d.]+)\s*:.+?:\s*([\d.]+)\s*<e>"

        # Find all matching timestamp pairs in the text
        matches = re.findall(timestamp_pattern, text)

        # Calculate midpoint timestamps and convert to strings
        # This gives us a more accurate representation of when the caption appears
        return [str((float(start) + float(end)) / 2) for start, end in matches]

    def _prepare_prompt(self, query: str, texts: List[str], images) -> str:
        context = "\n".join(texts[:3])  # Use top 3 most relevant text chunks

        return f""" 
            Analyze the video content and respond to the query provided below:  
            **Query:** {query}  

            ---

            ### Context from Video  
            {context}  

            ---

            ### Instructions  

            1. **Focus Areas**:  
            - Analyze both **visual** (e.g., images, frames) and **textual** (e.g., captions, transcriptions) elements present in the video.  

            2. **Timestamp Conversion**:  
            - Extract timestamps from frame image names based on the `frame_interval` value defined in the configuration YAML file.  
            - Convert timestamps to the MM:SS format (e.g., 125 seconds to 2:05).  
            - Expand each timestamp into a ±20-second range (e.g., 125s becomes 1:45 - 2:25).  

            3. **Output Format Requirements**:  
            - List all timestamp ranges at the end of your analysis in the following format:  
                `[Original Time] → [Start Window - End Window]`.  

            ---

            ### Include in the Analysis  

            - **Key Observations**:  
            - Highlight specific **frames** or **captions** that support your analysis.  

            - **Relevant Visual/Textual Evidence**:  
            - Identify and detail critical elements (e.g., objects, text on-screen, colors, emotions, actions) that substantiate your findings.  

            - **Relevant Timestamp Windows**:  
            - Ensure all timestamps are accurately converted to MM:SS format and expanded into ±20-second ranges.  

            ---

            ### Example Output Format  

            Main analysis content...  

            **Relevant Timestamp Windows:**  
            - 3:45 to 3:25 - 4:05  
            - 1:10 to 0:50 - 1:30  

            ---

            ### Configuration Information  
            - **Frame Interval**: 5 seconds  

            ---

            **Answer:**  """

    def process_query(
        self, retrieved_images: List[Path], retrieved_texts: List[str], query: str
    ) -> Dict[str, Any]:
        try:
            self.logger.info(f"Processing query: {query}")

            # Prepare images
            images = []
            for img_path in retrieved_images[
                :5
            ]:  # Limit to 5 images to avoid token limits
                try:
                    img = Image.open(str(img_path))
                    images.append(img)
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_path}: {e}")

            # Prepare prompt
            prompt = self._prepare_prompt(query, retrieved_texts, retrieved_images)

            # Generate response
            response = self.model.generate_content([prompt] + images)

            # Extract timestamps from all texts
            all_timestamps = []
            for text in retrieved_texts:
                all_timestamps.extend(self._extract_timestamps(text))

            # Format response
            result = {
                "answer": response.text,
                "source_images": [str(path) for path in retrieved_images[:5]],
                "timestamps": sorted(set(all_timestamps), key=float),
            }

            self.logger.info("Successfully processed query")
            return result

        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            raise
