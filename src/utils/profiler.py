"""
Performance Profiling for Edge CCTV Processing.

This module measures and logs per-frame latency and memory usage for each
pipeline stage during edge inference. It provides visibility into the
computational cost of processing CCTV video streams on resource-constrained
devices.

Typical usage:
    profiler = EdgeProfiler()
    profiler.start_stage("captioning")
    # ... run captioning ...
    profiler.end_stage("captioning")
    
    profiler.start_stage("verification")
    # ... run verification ...
    profiler.end_stage("verification")
    
    summary = profiler.get_summary()
    # summary contains avg latency and peak memory
"""

import time
import tracemalloc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

from src.utils.logger import Logger


logger = Logger(__name__)


@dataclass
class StageTiming:
    """Per-stage timing information."""
    stage_name: str
    timings_ms: List[float] = field(default_factory=list)
    
    def add_timing(self, timing_ms: float):
        """Record a timing measurement."""
        self.timings_ms.append(timing_ms)
    
    def avg_ms(self) -> float:
        """Get average timing in milliseconds."""
        return sum(self.timings_ms) / len(self.timings_ms) if self.timings_ms else 0.0
    
    def min_ms(self) -> float:
        """Get minimum timing in milliseconds."""
        return min(self.timings_ms) if self.timings_ms else 0.0
    
    def max_ms(self) -> float:
        """Get maximum timing in milliseconds."""
        return max(self.timings_ms) if self.timings_ms else 0.0


class EdgeProfiler:
    """
    Measure and log per-frame latency and memory usage for edge pipelines.
    
    This profiler tracks:
    - Latency (ms) for each pipeline stage per frame
    - Peak memory usage (MB) across the entire processing run
    - Frame throughput metrics
    
    Target deployment: Apple Silicon MPS on MacBook Pro for on-device
    CCTV video processing.
    
    Attributes:
        stages: Dict mapping stage names to StageTiming records
        current_stage: Currently active stage, or None
        stage_start_time: Start time of current stage
        peak_memory_mb: Peak memory usage observed
        total_frames_processed: Total frames processed
    """
    
    def __init__(self):
        """Initialize edge profiler."""
        self.stages: Dict[str, StageTiming] = {}
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
        self.peak_memory_mb: float = 0.0
        self.total_frames_processed: int = 0
        
        # Start memory tracking
        tracemalloc.start()
        logger.info("EdgeProfiler: Initialized (memory tracking enabled)")
    
    def start_stage(self, stage_name: str):
        """
        Start timing a pipeline stage.
        
        Args:
            stage_name: Name of the stage (e.g., "captioning", "verification")
        """
        if self.current_stage is not None:
            logger.warning(
                f"EdgeProfiler: Starting stage '{stage_name}' while '{self.current_stage}' "
                f"is still active. Call end_stage() first."
            )
        
        if stage_name not in self.stages:
            self.stages[stage_name] = StageTiming(stage_name)
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
    
    def end_stage(self, stage_name: Optional[str] = None) -> float:
        """
        End timing for a pipeline stage and record duration.
        
        Args:
            stage_name: Name of stage to end. If None, ends current stage.
        
        Returns:
            float: Elapsed time in milliseconds
        """
        if self.stage_start_time is None:
            logger.warning("EdgeProfiler: end_stage() called but no stage is active")
            return 0.0
        
        elapsed_ms = (time.time() - self.stage_start_time) * 1000
        
        # If stage_name provided, use it; otherwise use current
        active_stage = stage_name or self.current_stage
        
        if active_stage and active_stage in self.stages:
            self.stages[active_stage].add_timing(elapsed_ms)
        
        # Update peak memory
        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 ** 2)
        if peak_mb > self.peak_memory_mb:
            self.peak_memory_mb = peak_mb
        
        self.current_stage = None
        self.stage_start_time = None
        
        return elapsed_ms
    
    def record_frame(self):
        """Record that one frame has been processed."""
        self.total_frames_processed += 1
    
    def get_stage_stats(self, stage_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific stage.
        
        Args:
            stage_name: Name of the stage
        
        Returns:
            Dict with avg_ms, min_ms, max_ms, count
        """
        if stage_name not in self.stages:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "count": 0}
        
        stage = self.stages[stage_name]
        return {
            "avg_ms": stage.avg_ms(),
            "min_ms": stage.min_ms(),
            "max_ms": stage.max_ms(),
            "count": len(stage.timings_ms),
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive profiling summary.
        
        Returns:
            Dict with:
            - per_stage_stats: Dict mapping stage names to {"avg_ms", "min_ms", "max_ms"}
            - total_frames_processed: Total frames processed
            - peak_memory_mb: Peak memory usage in MB
            - avg_latency_per_frame_ms: Average total time per frame
        """
        # Calculate total latency per frame
        total_avg_latency = sum(
            stage.avg_ms() for stage in self.stages.values()
        )
        
        per_stage_stats = {}
        for stage_name, stage in self.stages.items():
            per_stage_stats[stage_name] = {
                "avg_ms": stage.avg_ms(),
                "min_ms": stage.min_ms(),
                "max_ms": stage.max_ms(),
            }
        
        return {
            "per_stage_stats": per_stage_stats,
            "total_frames_processed": self.total_frames_processed,
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "avg_latency_per_frame_ms": round(total_avg_latency, 2),
        }
    
    def save_summary(self, output_path: Path):
        """
        Save profiling summary to JSON file.
        
        Args:
            output_path: Path to save JSON summary
        """
        summary = self.get_summary()
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"EdgeProfiler: Saved summary to {output_path}")
        except Exception as e:
            logger.error(f"EdgeProfiler: Failed to save summary - {e}")
    
    def __del__(self):
        """Clean up memory tracking."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
