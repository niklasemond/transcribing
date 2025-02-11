# Congressional Hearing Transcriber

An advanced Python tool that automatically transcribes congressional hearing audio files, identifies speakers, and generates analytical summaries. Perfect for researchers, journalists, and policy analysts working with congressional hearing recordings.

## 🚀 Key Features
- 🎯 High-accuracy speech-to-text transcription using OpenAI Whisper
- 👥 Automatic speaker identification and separation using Pyannote
- 📊 Detailed analytical summaries with policy insights
- 📄 Multiple output formats (JSON, TXT, SRT)
- ⚡ Optimized for M1 Macs

## 🛠️ Installation

1. Clone the repository:
git clone https://github.com/niklas-emanuel/hearing-transcriber.git
cd hearing-transcriber

2. Install required packages:
pip install -r requirements.txt

3. Create a `.env` file with your tokens:
HF_TOKEN=your_huggingface_token

## 🔑 Prerequisites

1. **Hugging Face Account**:
   - Create account at [Hugging Face](https://huggingface.co)
   - Get token from https://huggingface.co/settings/tokens
   - Accept terms for:
     - [Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization)
     - [Segmentation](https://huggingface.co/pyannote/segmentation)

2. **System Requirements**:
   - Python 3.10+
   - ffmpeg (`brew install ffmpeg` on macOS)
   - Optimized for M1 Macs

## 💻 Usage

Run the optimized version:
python hearing_optimized.py path/to/your/audio.mp3

## 📊 Output Files
- `filename.json`: Full transcript with speaker labels and timestamps
- `filename.txt`: Readable transcript with analytical summary
- `filename.srt`: Subtitle format with speaker identification

## ⚙️ Processing Pipeline

1. **Audio Transcription**
   - Uses Whisper medium model
   - Optimized for English language
   - Includes timestamps

2. **Speaker Identification**
   - Automatically detects distinct speakers
   - Labels speaker segments
   - Handles overlapping speech

3. **Analytical Summary**
   - Main topics discussed
   - Key arguments presented
   - Notable speaker exchanges
   - Policy implications
   - Significant conclusions

## ⏱️ Processing Times
For a 5-minute audio file:
- Transcription: ~5-7 minutes
- Speaker ID: ~3-5 minutes
- Summary: ~1-2 minutes
- Total: ~9-14 minutes

## 🔄 Version Information
- `hearing.py`: Original version
- `hearing_optimized.py`: Enhanced version with improved performance and analytical summaries

## 📝 License
[MIT License](LICENSE)

## 🤝 Contributing
Contributions welcome! Feel free to:
- Open issues
- Submit pull requests
- Suggest improvements