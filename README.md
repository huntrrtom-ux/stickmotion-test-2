# Stickmotion — Voiceover Visualizer

Automatically generate paint-style stick figure visuals for any voiceover script. Upload audio, and the app will:

1. **Transcribe** the voiceover using OpenAI Whisper
2. **Detect scene changes** using GPT-4o to identify visual cues
3. **Generate animated video** (Veo) for the first 30 seconds
4. **Generate still illustrations** (Imagen) for remaining scenes with Ken Burns effect
5. **Compose** a final MP4 video with all visuals synced to the original audio

## Architecture

```
Audio Upload → Whisper Transcription → GPT-4o Scene Detection
    ↓
First 30s scenes → Google Veo (animated stick figure video)
Remaining scenes → Google Imagen (still stick figure illustrations)
    ↓
FFmpeg compositing → Final MP4 with audio
```

## Tech Stack

- **Backend**: Python / Flask / Flask-SocketIO
- **Transcription**: OpenAI Whisper API
- **Scene Analysis**: OpenAI GPT-4o
- **Image Generation**: Google Imagen 3
- **Video Generation**: Google Veo 2
- **Video Compositing**: FFmpeg
- **Deployment**: Docker / Railway

## Setup

### Prerequisites

- Python 3.11+
- FFmpeg installed on your system
- OpenAI API key (for Whisper + GPT-4o)
- Google AI API key (for Imagen + Veo)

### Local Development

```bash
# Clone and enter directory
cd voiceover-visualizer

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-your-key"
export GOOGLE_API_KEY="your-google-key"

# Run
python app.py
```

Visit `http://localhost:8080`

### Deploy to Railway

1. Push this repo to GitHub
2. Create a new project on [Railway](https://railway.app)
3. Connect your GitHub repo
4. Add environment variables in Railway dashboard:
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`
   - `SECRET_KEY` (optional)
5. Deploy — Railway will auto-detect the Dockerfile

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for Whisper and GPT-4o |
| `GOOGLE_API_KEY` | Yes | Google AI API key for Imagen and Veo |
| `SECRET_KEY` | No | Flask session secret (auto-generated if unset) |
| `PORT` | No | Server port (default: 8080, Railway sets automatically) |

## How It Works

### Scene Detection
GPT-4o analyzes the transcript with timestamps and identifies natural scene/topic changes. Each scene gets a detailed visual description tailored for stick figure illustration.

### Visual Style
All visuals maintain a consistent aesthetic:
- Hand-drawn paint/watercolor style
- Simple stick figure characters with round heads
- Warm cream/white backgrounds
- Children's storybook illustration feel

### Video vs. Still Images
- **First 30 seconds**: Animated via Google Veo for engaging opening
- **After 30 seconds**: Imagen still images with Ken Burns (slow zoom) effect for visual interest

### Fallback Behavior
If Veo generation fails, the app automatically falls back to generating a still image and converting it to video. If Imagen fails, a placeholder stick figure is drawn programmatically.

## Supported Audio Formats

MP3, WAV, OGG, FLAC, M4A, AAC, WebM, MP4 — up to 500MB

## Notes

- Processing time depends on audio length. A 2-minute voiceover typically takes 3-8 minutes.
- Veo video generation can take up to 5 minutes per scene.
- The Gunicorn timeout is set to 600 seconds to accommodate long processing times.
- WebSocket is used for real-time progress updates during processing.
