# Whisper Transcription Tool

A simple Python tool that uses OpenAI's Whisper API to transcribe audio files.

## Setup

1. Make sure you have [Poetry](https://python-poetry.org/) installed.
2. Make sure you have [ffmpeg](https://ffmpeg.org/) installed (required for audio processing).
3. Clone this repository.
4. Create a `.env` file in the root directory with your OpenAI API key:

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Install dependencies using Poetry:

   ```
   poetry install
   ```

## Usage

### Using the Shell Script (Recommended)

The easiest way to use this tool is with the provided shell script:

```bash
./transcribe /path/to/your/audio/file.mp3
```

This script automatically:

- Detects if your audio file is too long for direct API processing (over 25 minutes)
- For long files, it will split the audio into smaller chunks, transcribe each separately, and merge the results into a seamless transcript
- For shorter files, it will process them directly

You can pass any additional options after the file path:

```bash
./transcribe /path/to/your/audio/file.mp3 --model whisper-1
```

### Options

- `--model`: Choose the model to use for transcription (default: `gpt-4o-transcribe`)
  - Available options: `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`
- `--chunk-length`: Customize the length of each chunk in seconds for long audio files (default: 1200)
- `--keep-chunks`: Keep the intermediate audio chunks and their transcriptions

## Supported Audio Formats

- mp3
- mp4
- mpeg
- mpga
- m4a
- wav
- webm

## Output

The transcription is:

1. Displayed in the console (for files under the API limit)
2. Saved to a text file with the same name as the input audio file
3. For long files, the output appears as one continuous transcript with no visual indicators of the chunk boundaries

## File Size and Duration Limitations

The OpenAI API has limitations:

- File size: maximum 25 MB
- Duration: maximum 25 minutes (1500 seconds)

This tool automatically handles longer files by splitting them into smaller chunks while producing a single, cohesive transcription output.
