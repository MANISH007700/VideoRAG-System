import logging
from pathlib import Path
from typing import List, Tuple

from llama_index.core.schema import ImageNode


class VideoRetriever:
    def __init__(self, index):
        self.index = index
        self.retriever_engine = self.index.as_retriever(
            similarity_top_k=5, image_similarity_top_k=5
        )
        self.logger = logging.getLogger(__name__)

    def retrieve(self, query_str: str) -> Tuple[List[Path], List[str]]:
        try:
            self.logger.info(f"Processing query: {query_str}")
            retrieval_results = self.retriever_engine.retrieve(query_str)

            retrieved_images = []
            retrieved_texts = []

            for res_node in retrieval_results:
                if isinstance(res_node.node, ImageNode):
                    retrieved_images.append(Path(res_node.node.metadata["file_path"]))
                else:
                    retrieved_texts.append(res_node.text)

            self.logger.info(
                f"Retrieved {len(retrieved_images)} images and {len(retrieved_texts)} text segments"
            )
            return retrieved_images, retrieved_texts

        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise
