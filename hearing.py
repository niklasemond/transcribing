import argparse
import whisper
import json
import openai
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Load Pyannote speaker diarization model (Requires Hugging Face login)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                  use_auth_token=os.getenv("HF_TOKEN"))

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file):
    """Transcribes audio using OpenAI Whisper."""
    print("Loading Whisper model (this may take a minute)...")
    # Changed from "large" to "medium" for better speed/accuracy balance
    model = whisper.load_model("medium")
    
    print(f"\nTranscribing {audio_file}")
    print("This may take a while depending on the file length.")
    print("Processing... (You'll see results when complete)")
    
    # Add optimized parameters for faster processing
    result = model.transcribe(
        audio_file,
        fp16=False,  # Better for M1 CPU
        language='en',  # Specify English for faster processing
        initial_prompt="This is a congressional hearing transcript."  # Help with context
    )
    
    print("✓ Transcription complete!")
    
    transcript = []
    for segment in result["segments"]:
        transcript.append({
            "text": segment["text"],
            "start_time": segment["start"],
            "end_time": segment["end"]
        })

    return transcript

def diarize_audio(audio_file):
    """Performs speaker diarization (who said what)."""
    diarization = pipeline(audio_file)
    speakers = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((turn.start, turn.end))

    return speakers

def match_speakers(transcript, speakers):
    """Matches speakers to transcribed text using timestamps."""
    labeled_transcript = []

    for segment in transcript:
        start, end = segment["start_time"], segment["end_time"]
        speaker_id = "Unknown"

        for speaker, times in speakers.items():
            for (s_start, s_end) in times:
                if s_start <= start <= s_end:
                    speaker_id = speaker
                    break

        labeled_transcript.append({
            "speaker": speaker_id,
            "text": segment["text"],
            "start_time": start
        })

    return labeled_transcript

def generate_summary(transcript):
    """Summarizes the transcript using BART, optimized for M1 Mac."""
    from transformers import pipeline
    import torch

    text_data = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])
    
    # Initialize the summarizer with M1 optimization
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device="mps" if torch.backends.mps.is_available() else "cpu"  # Use M1's GPU if available
    )
    
    # Split long text into chunks (BART has a token limit)
    max_chunk = 1024
    chunks = [text_data[i:i + max_chunk] for i in range(0, len(text_data), max_chunk)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

def save_transcript(transcript, summary, audio_file):
    """Saves transcript in JSON, TXT, and SRT formats."""
    base_name = os.path.splitext(audio_file)[0]

    # Save JSON
    with open(f"{base_name}.json", "w") as json_file:
        json.dump({"transcript": transcript, "summary": summary}, json_file, indent=4)

    # Save TXT
    with open(f"{base_name}.txt", "w") as txt_file:
        for entry in transcript:
            txt_file.write(f"{entry['speaker']}: {entry['text']}\n")
        txt_file.write("\nSUMMARY:\n" + summary)

    # Save SRT (Subtitles)
    with open(f"{base_name}.srt", "w") as srt_file:
        for i, entry in enumerate(transcript, 1):
            start_s = entry["start_time"]
            end_s = start_s + 5  # Approximate subtitle duration
            srt_file.write(f"{i}\n")
            srt_file.write(f"{format_time(start_s)} --> {format_time(end_s)}\n")
            srt_file.write(f"{entry['speaker']}: {entry['text']}\n\n")

def format_time(seconds):
    """Converts seconds to SRT timestamp format."""
    hours, rem = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def main():
    parser = argparse.ArgumentParser(description="Transcribe and analyze congressional hearing audio.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (MP3/WAV).")
    args = parser.parse_args()

    print("\n🎯 Starting transcription process...")
    print(f"Audio file: {args.audio_file}")
    
    print("\n📝 Step 1: Transcribing audio...")
    transcript = transcribe_audio(args.audio_file)

    print("\n🎙️ Step 2: Identifying speakers...")
    speakers = diarize_audio(args.audio_file)
    labeled_transcript = match_speakers(transcript, speakers)

    print("\n🤖 Step 3: Summarizing transcript with BART...")
    summary = generate_summary(labeled_transcript)

    print("\n💾 Step 4: Saving transcript in multiple formats...")
    save_transcript(labeled_transcript, summary, args.audio_file)

    print("\n✅ Transcription complete! Files saved in the same directory.")

if __name__ == "__main__":
    main()