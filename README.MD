Broll Video Understanding Pipeline — Speed Optimization


Problem Statement

Before Broll can edit a video, it must first "watch and understand" the input footage. Historically, processing 1 hour of raw footage required ~2 hours of GPU compute time, creating the primary performance bottleneck across the entire editing pipeline.  
PDF
+ 1

The video understanding workflow consists of 3 sequential stages:  
PDF

Context Analysis (Vision): Identifies scene cuts, visual actions, and on-screen content.  
PDF

Semantic Understanding (Audio): Extracts speech transcripts, tone, and semantic meaning.  
PDF

Editorial Comprehension (LLM): Evaluates footage quality, identifying good takes, bad takes, and key edit moments.  
PDF

⏱️ Profiling Metrics & Bottleneck Analysis
Rather than guessing, execution profiling was conducted to measure the exact latency contribution of each stage.  
PDF

Live Profiling Metrics
======================================================================
                         PROFILING METRICS
======================================================================
1. Context Analysis (Vision)      :   2.33s  |   31.2%
2. Semantic Understanding (Audio) :   3.93s  |   52.6%
3. Editorial Comprehension (LLM)  :   1.21s  |   16.2%
----------------------------------------------------------------------
TOTAL LATENCY                     :   7.47s  |  100.0%
======================================================================
Key Finding
Semantic Understanding (Audio) is the primary bottleneck, consuming 52.6% of total execution time (3.93s out of 7.47s).

Context Analysis (Vision) accounts for 31.2% (2.33s).

Editorial Comprehension (LLM) accounts for 16.2% (1.21s).

🛠️ Stack & Optimization Techniques
To optimize the primary bottleneck without degrading editorial output quality, the following architecture was implemented:

Silero VAD (Voice Activity Detection) Gatekeeper: Short-circuits non-speech audio segments prior to hitting heavy CUDA/Whisper transcription models.

Groq LPU Acceleration: Replaces standard local GPU LLM execution with Groq LPUs for rapid semantic audio interpretation and editorial reasoning.  
PDF

Streamlit Executive Dashboard: Provides an interactive side-by-side comparative benchmark interface.

FFmpeg / Python Subprocess: Handles lightweight local frame and audio extraction.

📁 Repository Structure
Plaintext
Broll-Video-Faster/
├── .env                       # API keys (Excluded via .gitignore)
├── .gitignore                 # Specifies untracked binaries/videos
├── app.py                     # Streamlit 3D interactive dashboard
├── pipeline_test.py           # Terminal profiling profiler script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
⚙️ Installation & Setup
1. Environment Setup
PowerShell
# Clone repository
git clone https://github.com/Sharathweb/Broll-Video-Faster.git
cd Broll-Video-Faster

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
2. Environment Variables
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
🎬 Working with Sample Media
Large video files (.mp4, .mov) are excluded from Git tracking via .gitignore to prevent repository bloat and transfer timeouts.

Option A: Use Default Sample Video (Sample Video.mp4)
Place any local 1-minute test clip named Sample Video.mp4 in the project root folder. Alternatively, download a sample clip directly via PowerShell/curl:

PowerShell
curl -o "Sample Video.mp4" "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

Option B: Upload Custom B-Roll Video
The Streamlit interface includes an interactive file uploader supporting .mp4, .mov, and .avi files (up to 200MB).

🚀 Running the Benchmarks
1. Executive Dashboard (Streamlit UI)
Launch the side-by-side visual benchmarking dashboard:

PowerShell
streamlit run app.py
Navigate to http://localhost:8501.

Select Use Sample Video (Sample Video.mp4) or Upload Custom B-Roll Video in the sidebar.

Click Execute Comparative Benchmark to view latency profiles and side-by-side quality verification.

2. Terminal Profiler CLI
Run the profiling script directly in the terminal to output the raw latency table:

PowerShell
python pipeline_test.py