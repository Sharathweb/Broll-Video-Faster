"""
Video Benchmark Pipeline Profiler
Author: Sharath
Description: Measures runtime latency breakdown across Context Analysis (SGLang Vision), 
             Semantic Understanding (faster-whisper C++), and LLM Editorial Reasoning 
             to locate performance bottlenecks with minimal token usage and exact cut precision.
"""

import os
import time
import gc
import json
import logging
import subprocess
import warnings
import numpy as np
import torch
import cv2
import base64
import tempfile
import re
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel

load_dotenv()

# Suppress warnings cluttering stdout
warnings.filterwarnings("ignore", category=UserWarning)

# Torch / HF Caches to D: Drive
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_CACHE"] = r"D:\huggingface_cache\hub"
os.environ["TORCH_HOME"] = r"D:\torch_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PipelineProfiler")


def sync_gpu() -> None:
    """Synchronizes CUDA streams to ensure exact execution timing during profiling."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class PipelineProfiler:
    def __init__(self, video_path: str):
        self.video_path = video_path
        load_dotenv()
        self.device = "cuda" if self._has_cuda() else "cpu"
        
        # High-Efficiency SGLang Engine Initialization (Qwen2.5-VL via FP8/INT4)
        self.sglang_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.vision_model = "gpt-4o-mini"

        compute_type = "float16" if self.device == "cuda" else "int8"
        self.whisper_model = WhisperModel(
            "tiny.en",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            local_files_only=True  # Avoid HF network checks
        )

    def _has_cuda(self) -> bool:
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _extract_scene_keyframes(self, max_frames: int = 3, threshold: float = 28.0) -> List[Tuple[float, str]]:
        """
       Fast keyframe extractor using timestamp seek rather than full sequence read.
        """
        cap = cv2.VideoCapture(self.video_path)
        keyframes = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if total_frames > 0 else 0.0

            if duration_sec <= 0:
                timestamps = [0.0]
            else:
                timestamps = [
                    round((duration_sec / (max_frames + 1)) * i, 2)
                    for i in range(1, max_frames + 1)
                ]

            for ts in timestamps:
                frame_idx = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Resize to lower resolution (640x360) to keep base64 payload small
                    resized = cv2.resize(frame, (640, 360))
                    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    base64_str = base64.b64encode(buffer).decode('utf-8')
                    keyframes.append((ts, base64_str))

        finally:
            cap.release()

        return keyframes

    def profile_vision(self) -> Tuple[float, Dict[str, Any]]:
        """Step 1: Step 1: Analyzes visual scene keyframes using Groq Vision Engine for structured context."""
        logger.info("[1/3] Benchmarking Context Analysis (SGLang Vision Engine)...")
        sync_gpu()
        start_time = time.perf_counter()

        try:
            extracted_keyframes = self._extract_scene_keyframes(max_frames=3)

            if not extracted_keyframes:
                raise ValueError("Could not extract any valid scene keyframes from video.")

            content_payload = []
            for ts, b64 in extracted_keyframes:
                content_payload.append({
                    "type": "text",
                    "text": f"Frame timestamp: {ts}s"
                })
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })

            content_payload.append({
                "type": "text",
                "text": (
                    "Do NOT use thinking tags or internal reasoning.\n"
                    "Analyze these keyframes and return a valid JSON object matching exactly this structure:\n"
                    "{\n"
                    '  "scene_cuts": [\n'
                    '    {"timestamp_sec": 0.0, "description": "Scene overview"}\n'
                    "  ]\n"
                    "}"
                )
            })

            completion = self.sglang_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video profiling AI. Output strictly valid JSON."
                    },
                    {
                        "role": "user",
                        "content": content_payload
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )

            raw_content = completion.choices[0].message.content or ""
            cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

            json_match = re.search(r'\{.*\}', cleaned_content, flags=re.DOTALL)
            if json_match:
                vision_out = json.loads(json_match.group(0))
            else:
                raise ValueError(f"No JSON object found in model output: '{raw_content}'")
            
            json_str = json_match.group(0)
            sync_gpu()
            vision_out = json.loads(json_str)

        except Exception as err:
            logger.warning(f"SGLang Vision step fallback triggered: {err}")
            vision_out = {"scene_cuts": [{"timestamp_sec": 0.0, "description": "Visual scene cut fallback."}]}

        elapsed = time.perf_counter() - start_time
        logger.info(f"Vision analysis completed in {elapsed:.2f}s")
        return elapsed, vision_out

    def _fast_silero_vad_check(self) -> bool:
        """Extracts audio via FFmpeg and uses lightweight Silero VAD in Torch to verify speech presence."""
        cmd = [
            "ffmpeg", "-v", "error", "-i", self.video_path,
            "-f", "s16le", "-ac", "1", "-ar", "16000", "-"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_pcm, _ = process.communicate()

        if not raw_pcm or len(raw_pcm) == 0:
            return False

        audio_data = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 0.001:
            return False

        wav_tensor = torch.from_numpy(audio_data)

        try:
            model, utils = torch.hub.load(
                repo_or_dir=r"D:\torch_cache\hub\snakers4_silero-vad_master",
                source="local",
                model="silero_vad",
                trust_repo=True
            )
            get_speech_timestamps = utils[0]

            speech_timestamps = get_speech_timestamps(
                wav_tensor,
                model,
                threshold=0.2,
                sampling_rate=16000
            )
            return len(speech_timestamps) > 0
        except Exception as e:
            logger.warning(f"Silero VAD check error ({e}). Defaulting to proceeding with transcription.")
            return True

    def profile_audio(self) -> Tuple[float, str]:
        """Step 2: Transcribes audio stream using C++ faster-whisper large-v3-turbo."""
        logger.info("[2/3] Benchmarking Audio Understanding (faster-whisper C++)...")
        start_time = time.perf_counter()

        temp_audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.mp3")

        try:
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", self.video_path,
                    "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k",
                    temp_audio_path
                ]
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                # Upload compressed MP3 (typically under 15 MB for a 1-hour video)
                with open(temp_audio_path, "rb") as file:
                    transcription = self.sglang_client.audio.transcriptions.create(
                        file=(os.path.basename(temp_audio_path), file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                    )

                transcript_list = []
                segments = getattr(transcription, "segments", [])
                for segment in segments:
                    start = segment.get("start") if isinstance(segment, dict) else getattr(segment, "start", 0)
                    end = segment.get("end") if isinstance(segment, dict) else getattr(segment, "end", 0)
                    text = segment.get("text") if isinstance(segment, dict) else getattr(segment, "text", "")
            
                    if text.strip():
                        transcript_list.append(f"[{round(start, 1)}s-{round(end, 1)}s]: {text.strip()}")

                transcript = "\n".join(transcript_list) if transcript_list else "No spoken dialogue detected."

        except Exception as err:
            logger.warning(f"Audio step fallback triggered due to error: {err}")
            transcript = "Audio transcription fallback sample text."
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        elapsed = time.perf_counter() - start_time
        logger.info(f"Audio processing completed in {elapsed:.2f}s")
        return elapsed, transcript

    def profile_editorial(self, vision_context: Dict[str, Any], transcript_context: str) -> Tuple[float, Dict[str, Any]]:
        """Step 3: Evaluates multimodal context using SGLang text reasoning for structured JSON edit plan."""
        logger.info("[3/3] Benchmarking Editorial Comprehension...")
        sync_gpu()
        start_time = time.perf_counter()

        try:
            prompt = f"""
            You are a video editor AI. You must output raw JSON only matching this schema:
{{
  "edit_plan": [
    {{"start_sec": 0.0, "end_sec": 5.0, "action": "keep", "reason": "Intro sequence"}}
  ]
}}

Visual Cuts Context: {json.dumps(vision_context)}
Verbatim Audio Transcript: "{transcript_context}"
Respond ONLY with a valid JSON object.
            """

            completion = self.sglang_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
        {
            "role": "system",
            "content": "You are an expert video editor. You must respond strictly in JSON format." # <-- Must contain the word 'json'
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            plan = json.loads(completion.choices[0].message.content)
            sync_gpu()

        except Exception as err:
            logger.warning(f"LLM Reasoning step failed: {err}")
            plan = {"edit_plan": [{"start_sec": 0.0, "end_sec": 5.0, "action": "keep", "reason": "Fallback decision."}]}

        elapsed = time.perf_counter() - start_time
        logger.info(f"Editorial reasoning completed in {elapsed:.2f}s")
        return elapsed, plan

    def run_benchmark(self):
        if not os.path.exists(self.video_path):
            logger.error(f"Target video not found at path: '{self.video_path}'")
            return

        print("=" * 65)
        print(f" HIGH-EFFICIENCY PIPELINE BENCHMARK | Target: {self.video_path}")
        print("=" * 65)

        t_vis, vision_out = self.profile_vision()
        t_aud, audio_out = self.profile_audio()
        t_llm, edit_out = self.profile_editorial(vision_out, audio_out)

        total_time = t_vis + t_aud + t_llm
        p_vis = (t_vis / total_time * 100) if total_time else 0
        p_aud = (t_aud / total_time * 100) if total_time else 0
        p_llm = (t_llm / total_time * 100) if total_time else 0

        print("\n" + "=" * 65)
        print("                       PROFILING METRICS                       ")
        print("=" * 65)
        print(f" 1. Context Analysis (SGLang Vision): {t_vis:7.2f}s  |  {p_vis:5.1f}%")
        print(f" 2. Audio Understanding (faster-w): {t_aud:7.2f}s  |  {p_aud:5.1f}%")
        print(f" 3. Editorial Comprehension (LLM):   {t_llm:7.2f}s  |  {p_llm:5.1f}%")
        print("-" * 65)
        print(f" TOTAL LATENCY                  : {total_time:7.2f}s  |  100.0%")
        print("=" * 65)


if __name__ == "__main__":
    profiler = PipelineProfiler(video_path="Sample Video.mp4")
    profiler.run_benchmark()