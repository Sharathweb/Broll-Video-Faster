"""
Video Benchmark Pipeline Profiler
Author: Sharath
Description: Measures runtime latency breakdown across Vision (Groq Vision), Audio (WhisperX), 
             and LLM Editorial Reasoning (Groq) to locate performance bottlenecks.
"""

import os
import time
import gc
import logging
import subprocess
import warnings
import numpy as np
import torch
import cv2
import base64
import whisperx
from groq import Groq
from typing import Tuple, List
from dotenv import load_dotenv

# Suppress Pyannote/Torchcodec warnings cluttering stdout
warnings.filterwarnings("ignore", category=UserWarning)

# Because of Memory issue Hugging Face and PyTorch Caches to D: Drive
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_CACHE"] = r"D:\huggingface_cache\hub"
os.environ["TORCH_HOME"] = r"D:\torch_cache"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

# Preventing WhisperX / Hugging Face / Torch Hub from making unnecessary network calls
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
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Active Groq endpoints
        self.vision_model = "qwen/qwen3.6-27b"
        self.text_model = "llama-3.3-70b-versatile"

    def _has_cuda(self) -> bool:
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _extract_keyframes(self, num_frames: int = 3) -> List[str]:
        """Extracts linearly spaced keyframes. Capped at 3 frames max for Groq Vision limits."""

        cap = cv2.VideoCapture(self.video_path)
        base64_frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {self.video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Video contains no readable frames.")

            num_frames = min(num_frames, 3)
            step = max(1, total_frames // num_frames)
            indices = [min(i * step, total_frames - 1) for i in range(num_frames)]

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    resized = cv2.resize(frame, (1280, 720))
                    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    base64_str = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(base64_str)
        finally:
            cap.release()

        return base64_frames

    def profile_vision(self) -> Tuple[float, str]:
        """Step 1: Analyzes key video frames with Groq Multimodal Vision for visual context."""
        logger.info("[1/3] Benchmarking Context Analysis (Groq Vision API)...")
        start_time = time.perf_counter()

        try:
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is missing.")

            client = Groq(api_key=self.groq_api_key)
            base64_frames = self._extract_keyframes(num_frames=3)

            if not base64_frames:
                raise ValueError("Could not extract any valid frames from video.")

            content_payload = []
            for b64 in base64_frames:
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })

            content_payload.append({
                "type": "text",
                "text": "Identify main visual objects, scene framing, cuts, and key visual actions across these video frames."
            })

            completion = client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0.2,
                max_tokens=256
            )

            vision_summary = completion.choices[0].message.content

        except Exception as err:
            logger.warning(f"Groq Vision step failed: {err}")
            vision_summary = "Visual summary fallback: Multiple scene cuts detected."

        elapsed = time.perf_counter() - start_time
        logger.info(f"Vision analysis completed in {elapsed:.2f}s")
        return elapsed, vision_summary

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

        # Converting to torch tensor float32
        audio_data = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        wav_tensor = torch.from_numpy(audio_data)

        # Load Silero VAD from cached local model repository
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
            threshold=0.5, 
            sampling_rate=16000
        )

        return len(speech_timestamps) > 0

    def profile_audio(self) -> Tuple[float, str]:
        """Step 2: Transcribes audio stream using fast Silero VAD pre-screening and WhisperX."""
        logger.info("[2/3] Benchmarking Semantic Understanding (WhisperX)...")
        sync_gpu()
        start_time = time.perf_counter()

        try:
            has_speech = False
            try:
                has_speech = self._fast_silero_vad_check()
            except Exception as vad_err:
                logger.warning(f"Local Silero VAD check fallback (proceeding to WhisperX): {vad_err}")
                has_speech = True

            if not has_speech:
                logger.info("Silero VAD Gatekeeper: No speech timestamps detected in audio. Short-circuiting WhisperX.")
                elapsed = time.perf_counter() - start_time
                return elapsed, "No spoken dialogue detected in footage."

            

            compute_type = "float16" if self.device == "cuda" else "int8"

            whisper_model = whisperx.load_model(
                "small", 
                self.device, 
                compute_type=compute_type,
                language="en",
                vad_method="silero"
            )

            audio = whisperx.load_audio(self.video_path)
            raw_result = whisper_model.transcribe(audio, batch_size=16, language="en")

            segments = raw_result.get("segments", [])

            if not segments:
                transcript = "No spoken dialogue detected in footage."
            else:
                model_a, metadata = whisperx.load_align_model(
                    language_code="en", device=self.device
                )
                aligned_result = whisperx.align(
                    segments,
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
                transcript = " ".join([seg.get("text", "") for seg in aligned_result["segments"]])
                del model_a

            sync_gpu()

            del whisper_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as err:
            logger.warning(f"Audio step fallback triggered due to error: {err}")
            transcript = "Audio transcription fallback sample text."

        elapsed = time.perf_counter() - start_time
        logger.info(f"Audio processing completed in {elapsed:.2f}s")
        return elapsed, transcript

    def profile_editorial(self, vision_context: str, transcript_context: str) -> Tuple[float, str]:
        """Step 3: Evaluates multimodal context using Groq LLM to formulate an edit strategy."""
        logger.info("[3/3] Benchmarking Editorial Comprehension (Groq API)...")
        sync_gpu()
        start_time = time.perf_counter()

        try:
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is not defined in .env file.")

            client = Groq(api_key=self.groq_api_key)
            prompt = f"""
            Analyze raw footage metadata and output editing decisions:
            - Visual Scene Summary: {vision_context}
            - Audio Transcript: {transcript_context}

            Task: Identify core highlights, strip dead space/silence, and output timestamp cuts.
            """

            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.text_model,
                temperature=0.3,
                max_tokens=300
            )
            plan = completion.choices[0].message.content
            sync_gpu()

        except Exception as err:
            logger.warning(f"LLM Reasoning step failed: {err}")
            plan = "Editorial decision fallback plan."

        elapsed = time.perf_counter() - start_time
        logger.info(f"Editorial reasoning completed in {elapsed:.2f}s")
        return elapsed, plan

    def run_benchmark(self):
        if not os.path.exists(self.video_path):
            logger.error(f"Target video not found at path: '{self.video_path}'")
            return

        print("=" * 65)
        print(f" PIPELINE PERFORMANCE BENCHMARK | Target: {self.video_path}")
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
        print(f" 1. Context Analysis (Vision)   : {t_vis:7.2f}s  |  {p_vis:5.1f}%")
        print(f" 2. Semantic Understanding (Audio): {t_aud:7.2f}s  |  {p_aud:5.1f}%")
        print(f" 3. Editorial Comprehension (LLM): {t_llm:7.2f}s  |  {p_llm:5.1f}%")
        print("-" * 65)
        print(f" TOTAL LATENCY                  : {total_time:7.2f}s  |  100.0%")
        print("=" * 65)

        stages = [
            ("Vision (Groq Vision)", t_vis),
            ("Audio (WhisperX)", t_aud),
            ("LLM (Groq)", t_llm)
        ]
        bottleneck = max(stages, key=lambda item: item[1])
        print(f"\n[BOTTLENECK DETECTED] Stage '{bottleneck[0]}' dominated runtime ({bottleneck[1]:.2f}s).\n")


if __name__ == "__main__":
    profiler = PipelineProfiler(video_path="Sample Video.mp4")
    profiler.run_benchmark()