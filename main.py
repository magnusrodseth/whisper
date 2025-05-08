#!/usr/bin/env python3
"""
A script to transcribe audio files using the OpenAI Whisper API.
Automatically handles files longer than the API limit by splitting and merging.
Usage: python main.py /path/to/audio/file.mp3
"""

import os
import sys
import argparse
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# Maximum duration for direct API transcription (25 minutes = 1500 seconds)
MAX_DURATION = 1450  # Slightly below the limit for safety
DEFAULT_CHUNK_LENGTH = 1200  # 20 minutes (safe margin below limit)

def load_api_key():
    """Load OpenAI API key from .env file"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return api_key

def check_dependencies():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH. Please install ffmpeg.")
        sys.exit(1)

def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        float: Duration in seconds
    """
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def split_audio(input_file: str, chunk_length: int, output_dir: str) -> List[str]:
    """
    Split audio file into chunks of specified length.
    
    Args:
        input_file: Path to the input audio file
        chunk_length: Length of each chunk in seconds
        output_dir: Directory to store the chunks
        
    Returns:
        List of paths to the generated audio chunks
    """
    input_path = Path(input_file)
    filename_base = input_path.stem
    extension = input_path.suffix
    
    # Get audio duration
    duration = get_audio_duration(input_file)
    
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Splitting into chunks of {chunk_length} seconds...")
    
    chunk_files = []
    
    # Calculate number of chunks
    num_chunks = int(duration / chunk_length) + (1 if duration % chunk_length > 0 else 0)
    
    # Split audio into chunks
    for i in range(num_chunks):
        start_time = i * chunk_length
        output_file = os.path.join(output_dir, f"{filename_base}_chunk_{i+1:03d}{extension}")
        
        cmd = [
            "ffmpeg",
            "-i", input_file,
            "-ss", str(start_time),
            "-t", str(chunk_length),
            "-c", "copy",
            "-y",  # Overwrite output files without asking
            output_file
        ]
        
        print(f"Creating chunk {i+1}/{num_chunks}...")
        subprocess.run(cmd, capture_output=True, check=True)
        chunk_files.append(output_file)
    
    return chunk_files

def transcribe_audio(file_path, model="gpt-4o-transcribe"):
    """
    Transcribe audio using OpenAI's Whisper API
    
    Args:
        file_path (str): Path to the audio file
        model (str): Model to use for transcription (default: gpt-4o-transcribe)
        
    Returns:
        str: Transcribed text
    """
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check file extension
    supported_formats = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
    file_extension = file_path.suffix.lower().lstrip(".")
    if file_extension not in supported_formats:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
    
    print(f"Transcribing {file_path.name}...")
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

def transcribe_audio_with_split(file_path, model="gpt-4o-transcribe", chunk_length=DEFAULT_CHUNK_LENGTH, keep_chunks=False):
    """
    Handle transcription for audio files that exceed the API limits by splitting them.
    
    Args:
        file_path (str): Path to the audio file
        model (str): Model to use for transcription
        chunk_length (int): Length of each chunk in seconds
        keep_chunks (bool): Whether to keep the audio chunks after transcription
        
    Returns:
        str: Path to the output transcription file
    """
    input_path = Path(file_path)
    output_transcription = input_path.with_suffix(".txt")  # Use .txt for all files
    
    # Create temporary directory for chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory for chunks: {temp_dir}")
        
        try:
            # Split audio into chunks
            chunk_files = split_audio(file_path, chunk_length, temp_dir)
            
            # Transcribe each chunk and collect texts
            all_transcriptions = []
            
            for i, chunk_file in enumerate(chunk_files):
                print(f"Transcribing chunk {i+1}/{len(chunk_files)}...")
                try:
                    transcript = transcribe_audio(chunk_file, model)
                    all_transcriptions.append((i+1, transcript))
                except Exception as e:
                    print(f"Error transcribing chunk {i+1}: {e}")
            
            # Write merged transcription as one continuous text
            with open(output_transcription, 'w') as outfile:
                for chunk_num, text in all_transcriptions:
                    # Remove chunk markers, just append the text directly
                    outfile.write(text)
                    # Add a space between chunks to ensure they don't run together
                    outfile.write(" ")
            
            # Copy audio chunks to a directory if requested
            if keep_chunks:
                chunks_dir = input_path.parent / f"{input_path.stem}_chunks"
                os.makedirs(chunks_dir, exist_ok=True)
                
                for i, chunk_file in enumerate(chunk_files):
                    chunk_name = os.path.basename(chunk_file)
                    shutil.copy2(chunk_file, chunks_dir / chunk_name)
                    
                    # Create individual transcription files
                    chunk_text = next((text for num, text in all_transcriptions if num == i+1), "")
                    if chunk_text:
                        chunk_txt = (chunks_dir / chunk_name).with_suffix(".txt")
                        with open(chunk_txt, 'w') as f:
                            f.write(chunk_text)
                
                print(f"Audio chunks and individual transcriptions saved to: {chunks_dir}")
            
            return output_transcription
            
        except Exception as e:
            print(f"Error: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper API")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument(
        "--model", 
        default="whisper-1", 
        choices=["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        help="Model to use for transcription (default: whisper-1)"
    )
    parser.add_argument(
        "--chunk-length", 
        type=int, 
        default=DEFAULT_CHUNK_LENGTH,
        help=f"Length of each chunk in seconds (default: {DEFAULT_CHUNK_LENGTH})"
    )
    parser.add_argument(
        "--keep-chunks", 
        action="store_true",
        help="Keep the audio chunks after transcription (for long files)"
    )
    args = parser.parse_args()
    
    try:
        # Check if ffmpeg is installed
        check_dependencies()
        
        file_path = args.audio_file
        
        # Check audio duration
        duration = get_audio_duration(file_path)
        print(f"Audio duration: {duration:.2f} seconds")
        
        if duration > MAX_DURATION:
            print(f"Long audio detected (over {MAX_DURATION} seconds). Using split-transcribe method...")
            output_file = transcribe_audio_with_split(
                file_path, 
                args.model, 
                args.chunk_length, 
                args.keep_chunks
            )
            print(f"\nTranscription of long audio saved to: {output_file}")
        else:
            print("Audio within standard limits. Using direct transcription...")
            transcript = transcribe_audio(file_path, args.model)
            
            print("\nTranscription:")
            print("---------------")
            print(transcript)
            
            # Save transcript to a text file
            output_file = Path(file_path).with_suffix(".txt")
            with open(output_file, "w") as f:
                f.write(transcript)
            print(f"\nTranscription saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 