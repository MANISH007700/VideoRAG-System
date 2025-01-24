import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yt_dlp
from moviepy.video.io.VideoFileClip import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi


@dataclass
class VideoMetadata:
    title: str
    author: str
    views: int
    duration: int  # Video duration in seconds
    filesize_mb: float
    format: str
    resolution: str
    video_id: str


class VideoProcessor:
    def __init__(self, url: str, config: dict):
        self.url = url
        self.config = config
        self.video_id = self._extract_video_id(url)
        self.logger = logging.getLogger(__name__)

    def _extract_video_id(self, url: str) -> str:
        try:
            if "?v=" in url:
                return url.split("?v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            raise ValueError("Invalid YouTube URL format")
        except Exception as e:
            self.logger.error(f"Failed to extract video ID from URL: {url}")
            raise ValueError(f"Invalid YouTube URL: {e}")

    def download_video(self, progress_callback=None) -> Tuple[VideoMetadata, Path]:
        filename = f"{self.video_id}.mp4"
        output_path = Path(self.config["video_dir"]) / filename

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(output_path),
            "quiet": False,
            "no_warnings": True,
            "progress_hooks": [self._progress_hook],
        }

        self._progress_callback = progress_callback

        try:
            self.logger.info(f"Starting download of video: {self.url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)

            # Verify download
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Download failed - no file created at {output_path}"
                )

            # Get video duration
            with VideoFileClip(str(output_path)) as clip:
                duration = int(clip.duration)

            metadata = VideoMetadata(
                title=info.get("title", "Unknown"),
                author=info.get("uploader", "Unknown"),
                views=info.get("view_count", 0),
                duration=duration,
                filesize_mb=round(output_path.stat().st_size / (1024 * 1024), 2),
                format=info.get("format", "Unknown"),
                resolution=f"{info.get('width', 'Unknown')}x{info.get('height', 'Unknown')}",
                video_id=self.video_id,
            )
            return metadata, output_path
        except Exception as e:
            self.logger.error(f"Failed to download video: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            raise
        finally:
            self._progress_callback = None

    def _progress_hook(self, d):
        if self._progress_callback:
            status = ""
            if d["status"] == "downloading":
                # Simple status without calculations
                status = f"Downloading video... ({d.get('_percent_str', '')})"
            elif d["status"] == "finished":
                status = "Processing downloaded video..."

            if status:
                self._progress_callback(status)

    def extract_frames(self, video_path: Path) -> Path:
        output_dir = Path(self.config["data_dir"])
        output_dir.mkdir(exist_ok=True)

        try:
            self.logger.info(f"Extracting frames from video: {video_path}")
            with VideoFileClip(str(video_path)) as clip:
                fps = 1 / self.config["frame_interval"]
                clip.write_images_sequence(str(output_dir / "frame%04d.png"), fps=fps)
            return output_dir
        except Exception as e:
            self.logger.error(f"Failed to extract frames: {str(e)}")
            raise

    def extract_captions(self) -> Path:
        try:
            self.logger.info(f"Extracting captions for video: {self.video_id}")
            srt = YouTubeTranscriptApi.get_transcript(self.video_id)

            caption_file = (
                Path(self.config["data_dir"]) / f"captions_{self.video_id}.txt"
            )
            caption_text = []

            for entry in srt:
                start = entry["start"]
                end = start + entry["duration"]
                text = entry["text"].strip()
                caption_text.append(f"<s> {start:.2f} | {end:.2f} | {text} </s>")

            caption_file.write_text("\n".join(caption_text))
            self.logger.info(f"Captions saved with {len(caption_text)} entries")
            return caption_file
        except Exception as e:
            self.logger.error(f"Failed to extract captions: {str(e)}")
            raise
