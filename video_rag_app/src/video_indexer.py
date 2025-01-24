import logging
from pathlib import Path

import qdrant_client
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


class VideoIndexer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config["embed_model"]
        )

    def _index_exists(self, video_id: str) -> bool:
        try:
            client = qdrant_client.QdrantClient(
                path=str(Path(self.config["indexing_path"]))
            )
            collections = client.get_collections()
            expected_collections = {f"text_{video_id}", f"image_{video_id}"}
            existing_collections = {col.name for col in collections.collections}

            if all(col in existing_collections for col in expected_collections):
                for collection_name in expected_collections:
                    collection_info = client.get_collection(collection_name)
                    if collection_info.points_count == 0:
                        self.logger.warning(
                            f"Collection {collection_name} exists but is empty"
                        )
                        return False
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking index existence: {str(e)}")
            return False

    def create_multimodal_index(
        self, frames_dir: Path, captions_path: Path, video_id: str
    ) -> MultiModalVectorStoreIndex:
        try:
            self.logger.info("Creating new multimodal index...")

            client = qdrant_client.QdrantClient(
                path=str(Path(self.config["indexing_path"]))
            )

            # Create vector stores for text and images
            text_store = QdrantVectorStore(
                client=client, collection_name=f"text_{video_id}"
            )
            image_store = QdrantVectorStore(
                client=client, collection_name=f"image_{video_id}"
            )

            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=text_store, image_store=image_store
            )

            # Load documents (frames and captions)
            documents = SimpleDirectoryReader(str(frames_dir)).load_data()

            # Create index
            index = MultiModalVectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
            )

            self.logger.info("Successfully created multimodal index")
            return index

        except Exception as e:
            self.logger.error(f"Failed to create multimodal index: {str(e)}")
            raise

    def load_existing_index(self, video_id: str) -> MultiModalVectorStoreIndex:
        try:
            self.logger.info(f"Loading existing index for video {video_id}...")

            client = qdrant_client.QdrantClient(
                path=str(Path(self.config["indexing_path"]))
            )

            text_store = QdrantVectorStore(
                client=client, collection_name=f"text_{video_id}"
            )
            image_store = QdrantVectorStore(
                client=client, collection_name=f"image_{video_id}"
            )

            storage_context = StorageContext.from_defaults(
                vector_store=text_store, image_store=image_store
            )

            index = MultiModalVectorStoreIndex.from_vector_store(
                vector_store=text_store,
                storage_context=storage_context,
            )

            self.logger.info("Successfully loaded existing index")
            return index

        except Exception as e:
            self.logger.error(f"Failed to load existing index: {str(e)}")
            raise
