"""
Keyframe Extraction from Video Clips.

This module provides functionality to sample keyframes from video files
at configurable intervals. It is optimized for edge CCTV processing where
full-frame analysis would be prohibitively expensive.

The sampler extracts frames at a specified interval (e.g., every 3 seconds)
and returns them as PIL Images with associated timestamps.

Typical usage:
    sampler = KeyframeSampler(video_path="data/virat/clips/VIRAT_S_000000.mp4")
    keyframes = sampler.extract_keyframes(interval_seconds=3.0, max_frames=50)
    # keyframes is a list of (timestamp_seconds: float, image: PIL.Image)
"""

import cv2
import time
from pathlib import Path
from typing import List, Tuple, Union, Optional
from PIL import Image
import numpy as np

from src.utils.logger import Logger


logger = Logger(__name__)


class KeyframeSampler:
    """
    Extract keyframes from video clips at configurable intervals.
    
    This sampler is designed for CCTV video processing, where analyzing
    every frame would consume excessive computational resources. By sampling
    at intervals (e.g., 3 seconds), we capture the essential visual content
    while remaining computationally feasible on edge devices.
    
    Attributes:
        video_path: Path to input video file
        fps: Frames per second of the video
        total_frames: Total number of frames in video
        duration_seconds: Duration of video in seconds
    """
    
    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize keyframe sampler for a video file.
        
        Args:
            video_path: Path to input video file
            
        Raises:
            FileNotFoundError: If video file does not exist
            RuntimeError: If video cannot be opened
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            
            # Extract video metadata
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0
            
            logger.info(
                f"KeyframeSampler: Opened {self.video_path.name} "
                f"({self.total_frames} frames, {self.duration_seconds:.1f}s, {self.fps:.1f} fps)"
            )
            
        except Exception as e:
            logger.error(f"KeyframeSampler: Failed to initialize - {e}")
            raise RuntimeError(f"Failed to initialize video sampler: {e}")
    
    def extract_keyframes(
        self,
        interval_seconds: float = 3.0,
        max_frames: int = 50
    ) -> List[Tuple[float, Image.Image]]:
        """
        Extract keyframes at specified interval.
        
        Args:
            interval_seconds: Interval between keyframes in seconds (default 3.0)
            max_frames: Maximum number of frames to extract, prevents runaway
                       processing on very long clips (default 50)
        
        Returns:
            List of (timestamp_seconds, PIL.Image) tuples, where:
            - timestamp_seconds is the time of the frame in the video
            - PIL.Image is the RGB image frame
        
        Raises:
            ValueError: If interval_seconds <= 0 or max_frames <= 0
            RuntimeError: If frame extraction fails
        """
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0, got {interval_seconds}")
        if max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {max_frames}")
        
        keyframes: List[Tuple[float, Image.Image]] = []
        start_time = time.time()
        
        try:
            # Reset video to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Calculate frame interval
            frame_interval = int(interval_seconds * self.fps)
            if frame_interval < 1:
                frame_interval = 1
                logger.warning(
                    f"KeyframeSampler: interval_seconds={interval_seconds} is very small; "
                    f"sampling every frame"
                )
            
            current_frame = 0
            frames_extracted = 0
            
            while frames_extracted < max_frames:
                # Set frame position
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                ret, frame_bgr = self.cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Calculate timestamp
                timestamp = current_frame / self.fps if self.fps > 0 else 0.0
                
                keyframes.append((timestamp, pil_image))
                frames_extracted += 1
                
                # Move to next keyframe
                current_frame += frame_interval
            
            extraction_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"KeyframeSampler: Extracted {frames_extracted} keyframes "
                f"in {extraction_time_ms:.2f}ms "
                f"(interval={interval_seconds}s, fps={self.fps:.1f})"
            )
            
            return keyframes
            
        except Exception as e:
            logger.error(f"KeyframeSampler: Extraction failed - {e}")
            raise RuntimeError(f"Failed to extract keyframes: {e}")
    
    def __del__(self):
        """Clean up video capture resource."""
        if hasattr(self, 'cap'):
            self.cap.release()
