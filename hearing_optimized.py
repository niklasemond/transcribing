import argparse
import whisper
import json
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
from openai import OpenAI
import torch
from transformers import pipeline
import time

# Load environment variables from .env file
load_dotenv()

# Global model initialization (load models once)
print("Initializing models...")
WHISPER_MODEL = whisper.load_model("medium")
DIARIZATION_PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                              use_auth_token=os.getenv("HF_TOKEN"))
SUMMARIZER = pipeline("summarization", 
                     model="facebook/bart-large-cnn",
                     device="mps" if torch.backends.mps.is_available() else "cpu")

def transcribe_audio(audio_file):
    """Transcribes audio using OpenAI Whisper."""
    print(f"\nTranscribing {audio_file}")
    print("Processing... (You'll see results when complete)")
    
    result = WHISPER_MODEL.transcribe(
        audio_file,
        fp16=False,  # Better for M1 CPU
        language='en',  # Specify English for faster processing
        initial_prompt="This is a congressional hearing transcript.",  # Help with context
        verbose=False  # Reduce console output
    )
    
    print("✓ Transcription complete!")
    return [{"text": segment["text"],
             "start_time": segment["start"],
             "end_time": segment["end"]} for segment in result["segments"]]

def diarize_audio(audio_file):
    """Performs speaker diarization (who said what)."""
    start_time = time.time()
    print("\nProcessing speaker identification...")
    print("This step typically takes 3-5 minutes for a 5-minute audio file")
    print("Still processing... Please wait...")
    
    diarization = DIARIZATION_PIPELINE(audio_file)
    
    # Show active processing
    speakers = {}
    processed_time = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.setdefault(speaker, []).append((turn.start, turn.end))
        if turn.end > processed_time:
            processed_time = turn.end
            elapsed = time.time() - start_time
            print(f"\rProcessed {processed_time:.1f} seconds of audio... (Time elapsed: {elapsed:.1f}s)", end="")
    
    total_time = time.time() - start_time
    print(f"\n✓ Speaker identification complete! Time taken: {total_time:.1f} seconds")
    print(f"Found {len(speakers)} distinct speakers")
    return speakers

def match_speakers(transcript, speakers):
    """Matches speakers to transcribed text using timestamps."""
    labeled_transcript = []
    speaker_times = {speaker: times for speaker, times in speakers.items()}
    
    for segment in transcript:
        start = segment["start_time"]
        speaker_id = "Unknown"
        
        # More efficient speaker matching
        for speaker, times in speaker_times.items():
            if any(s_start <= start <= s_end for s_start, s_end in times):
                speaker_id = speaker
                break
                
        labeled_transcript.append({
            "speaker": speaker_id,
            "text": segment["text"],
            "start_time": start
        })
    
    return labeled_transcript

def generate_summary(transcript):
    """Generates an analytical summary of the congressional hearing."""
    client = OpenAI()
    
    text_data = "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)
    
    print("\nGenerating analytical summary...")
    
    prompt = """Please provide an analytical summary of this congressional hearing that includes:
    1. Main topics discussed
    2. Key arguments or positions presented
    3. Notable exchanges between speakers
    4. Important policy implications
    5. Any significant decisions or conclusions reached
    
    Format the summary with clear sections and bullet points where appropriate."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert political analyst summarizing congressional hearings."},
            {"role": "user", "content": f"{prompt}\n\nHearing transcript:\n{text_data}"}
        ]
    )
    
    return response.choices[0].message.content

def save_transcript(transcript, summary, audio_file):
    """Saves transcript in JSON, TXT, and SRT formats."""
    base_name = os.path.splitext(audio_file)[0]
    
    # Use context managers for file operations
    with open(f"{base_name}.json", "w") as json_file:
        json.dump({"transcript": transcript, "summary": summary}, json_file, indent=4)
    
    with open(f"{base_name}.txt", "w") as txt_file:
        txt_file.write("\n".join(f"{entry['speaker']}: {entry['text']}" for entry in transcript))
        txt_file.write("\n\nSUMMARY:\n" + summary)
    
    with open(f"{base_name}.srt", "w") as srt_file:
        for i, entry in enumerate(transcript, 1):
            srt_file.write(f"{i}\n")
            start_s = entry["start_time"]
            end_s = start_s + 5
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
    
    try:
        # If you want to use the existing transcript and just generate a new summary:
        with open(f"{os.path.splitext(args.audio_file)[0]}.json", "r") as f:
            data = json.load(f)
            labeled_transcript = data["transcript"]
        
        # Generate new analytical summary
        summary = generate_summary(labeled_transcript)
        
        # Save updated version
        save_transcript(labeled_transcript, summary, args.audio_file)
        print("\n✅ New analytical summary generated and saved!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())