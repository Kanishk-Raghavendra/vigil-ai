"""
VIGIL-Edge Gradio Demo Application.

Interactive web interface for real-time CCTV video hallucination detection.
Enables users to upload video clips, configure processing parameters, and
visualize per-frame trust decisions with temporal consistency analysis.

Features:
- Video file upload (MP4, AVI, MOV)
- Configurable keyframe interval and processing limits
- Selectable aggregation mode (Balanced / Safe)
- Per-frame claim visualization with color-coded trust indicators
- Temporal inconsistency highlighting
- Real-time latency and memory profiling
- Local processing, no API calls required

Typical usage:
    cd /path/to/vigil-ai
    python demo/app.py
    # Open http://localhost:7860 in browser
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, List
import json

from src.pipeline.vigil_video_pipeline import VIGILVideoOrcestrator
from src.utils.logger import Logger


logger = Logger(__name__)


class VIGILDemoApp:
    """Gradio interface for VIGIL video processing."""
    
    def __init__(self):
        """Initialize demo app with pipeline."""
        self.orchestrator = VIGILVideoOrcestrator()
        self.last_results = None
    
    def process_video_demo(
        self,
        video_file,
        interval_seconds: float,
        aggregation_mode: str
    ) -> Tuple[str, str, str]:
        """
        Process uploaded video and return results for visualization.
        
        Args:
            video_file: Uploaded video file
            interval_seconds: Keyframe interval in seconds
            aggregation_mode: "Balanced" or "Safe"
        
        Returns:
            Tuple of (summary_html, claims_html, profiler_html)
        """
        try:
            # Process video
            logger.info(f"DemoApp: Processing video with interval={interval_seconds}s")
            
            results = self.orchestrator.process_video(
                video_path=video_file,
                interval_seconds=interval_seconds,
                max_frames=50
            )
            
            self.last_results = results
            
            # Generate HTML summaries
            summary_html = self._generate_summary_html(results)
            claims_html = self._generate_claims_html(results)
            profiler_html = self._generate_profiler_html(results)
            
            return summary_html, claims_html, profiler_html
            
        except Exception as e:
            error_msg = f"<p style='color: red;'>Error: {str(e)}</p>"
            return error_msg, error_msg, error_msg
    
    def _generate_summary_html(self, results: dict) -> str:
        """Generate video summary HTML."""
        summary = results.get("video_summary", {})
        
        html = "<div style='background: #f0f0f0; padding: 20px; border-radius: 8px;'>"
        html += "<h2>Video Summary</h2>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        
        for key, value in summary.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            html += f"<tr style='border-bottom: 1px solid #ddd;'>"
            html += f"<td style='padding: 8px; font-weight: bold;'>{key}</td>"
            html += f"<td style='padding: 8px;'>{value}</td>"
            html += f"</tr>"
        
        html += "</table></div>"
        return html
    
    def _generate_claims_html(self, results: dict) -> str:
        """Generate claims visualization HTML."""
        frames = results.get("frames", [])
        
        html = "<div style='background: #f9f9f9; padding: 20px;'>"
        html += "<h2>Per-Frame Claims</h2>"
        
        for frame in frames:
            timestamp = frame.get("timestamp_seconds", 0.0)
            caption = frame.get("caption", "")
            claims = frame.get("claims", [])
            temporal_flags = frame.get("temporal_flags", [])
            
            html += f"<div style='margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 4px;'>"
            html += f"<h3>Frame @ {timestamp:.1f}s</h3>"
            html += f"<p><strong>Caption:</strong> {caption}</p>"
            
            # Claims table
            html += "<table style='width: 100%; border-collapse: collapse;'>"
            html += "<tr style='background: #e0e0e0;'>"
            html += "<th style='padding: 8px; border: 1px solid #ccc;'>Claim</th>"
            html += "<th style='padding: 8px; border: 1px solid #ccc;'>Score</th>"
            html += "<th style='padding: 8px; border: 1px solid #ccc;'>Decision</th>"
            html += "</tr>"
            
            for claim in claims:
                text = claim.get("text", "")
                score = claim.get("score", 0.0)
                decision = claim.get("decision", "UNKNOWN")
                
                # Color code by decision
                if decision == "TRUSTED":
                    row_color = "#e8f5e9"  # light green
                    decision_display = "✓ TRUSTED"
                elif decision == "REJECTED":
                    row_color = "#ffebee"  # light red
                    decision_display = "✗ REJECTED"
                else:
                    row_color = "#fff3e0"  # light yellow
                    decision_display = decision
                
                html += f"<tr style='background: {row_color};'>"
                html += f"<td style='padding: 8px; border: 1px solid #ccc;'>{text}</td>"
                html += f"<td style='padding: 8px; border: 1px solid #ccc;'>{score:.4f}</td>"
                html += f"<td style='padding: 8px; border: 1px solid #ccc;'>{decision_display}</td>"
                html += f"</tr>"
            
            html += "</table>"
            
            # Temporal flags
            if temporal_flags:
                html += "<div style='margin-top: 10px; padding: 10px; background: #fff8dc; border: 1px solid #f39c12;'>"
                html += "<strong>⚠ Temporal Inconsistencies:</strong><ul>"
                for flag in temporal_flags:
                    claim_text = flag.get("claim_text", "")
                    drop = flag.get("confidence_drop", 0.0)
                    html += f"<li>{claim_text} (confidence drop: {drop:.4f})</li>"
                html += "</ul></div>"
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _generate_profiler_html(self, results: dict) -> str:
        """Generate profiler summary HTML."""
        profiler = results.get("profiler_summary", {})
        
        html = "<div style='background: #f0f0f0; padding: 20px; border-radius: 8px;'>"
        html += "<h2>Performance Profiling</h2>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        
        # Per-stage stats
        per_stage = profiler.get("per_stage_stats", {})
        html += "<tr style='background: #e0e0e0;'><td colspan='4' style='padding: 8px; font-weight: bold;'>Stage Latencies (ms)</td></tr>"
        
        for stage_name, stats in per_stage.items():
            avg_ms = stats.get("avg_ms", 0.0)
            html += f"<tr style='border-bottom: 1px solid #ddd;'>"
            html += f"<td style='padding: 8px;'>{stage_name}</td>"
            html += f"<td style='padding: 8px;'>avg: {avg_ms:.2f}ms</td>"
            html += f"</tr>"
        
        # Overall stats
        html += "<tr style='background: #c8e6c9;'>"
        html += f"<td colspan='2' style='padding: 10px; font-weight: bold;'>Total</td>"
        html += f"<td style='padding: 10px;'>{profiler.get('avg_latency_per_frame_ms', 0):.2f}ms/frame</td>"
        html += f"<td style='padding: 10px;'>{profiler.get('peak_memory_mb', 0):.2f}MB peak</td>"
        html += f"</tr>"
        
        html += "</table></div>"
        return html


def launch_app():
    """Launch Gradio interface."""
    app = VIGILDemoApp()
    
    with gr.Blocks(title="VIGIL-Edge: CCTV Hallucination Detector") as interface:
        gr.Markdown("# VIGIL-Edge: Explainable On-Device CCTV Hallucination Detector")
        gr.Markdown(
            "Upload a CCTV video clip to detect hallucinations in vision-language "
            "model outputs. Processing happens entirely on-device with no internet required."
        )
        
        with gr.Row():
            # Left panel — inputs
            with gr.Column(scale=1):
                gr.Markdown("## Configuration")
                
                video_input = gr.File(
                    label="Upload Video (MP4, AVI, MOV)",
                    file_count="single",
                    type="filepath"
                )
                
                interval_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=0.5,
                    value=3.0,
                    label="Keyframe Interval (seconds)"
                )
                
                mode_selector = gr.Radio(
                    choices=["Balanced", "Safe"],
                    value="Balanced",
                    label="Mode"
                )
                
                run_button = gr.Button(
                    "🚀 Run Analysis",
                    variant="primary"
                )
            
            # Right panel — outputs
            with gr.Column(scale=2):
                gr.Markdown("## Results")
                
                summary_output = gr.HTML(
                    label="Summary",
                    value="<p>Upload a video and click Run Analysis</p>"
                )
                
                claims_output = gr.HTML(
                    label="Per-Frame Claims"
                )
                
                profiler_output = gr.HTML(
                    label="Performance Metrics"
                )
        
        # Connect button
        run_button.click(
            fn=app.process_video_demo,
            inputs=[video_input, interval_slider, mode_selector],
            outputs=[summary_output, claims_output, profiler_output]
        )
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    logger.info("Launching VIGIL-Edge Gradio demo app...")
    launch_app()
