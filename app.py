import os
import json
import time
import uuid
import tempfile
import subprocess
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import openai
from google import genai
from google.genai import types
import base64
import requests
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# API Clients
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'webm', 'mp4'}

# Consistent style prompt for all image generation
STYLE_PROMPT = (
    "Digital cartoon illustration with stick figure characters. "
    "Characters have large round white heads with simple dot eyes and expressive faces, "
    "thin black line bodies with simple hands and feet, wearing simple clothing. "
    "Rich detailed painted backgrounds with depth, texture, and atmospheric lighting. "
    "Style similar to web comics and animated explainer videos. "
    "Warm earthy color palette with greens, browns, and muted tones. "
    "High quality digital art, painterly textured background, cartoon style."
)

VIDEO_STYLE_PROMPT = (
    "Animated cartoon stick figure animation with expressive characters. "
    "Characters have large round white heads with simple facial expressions, "
    "thin black line bodies wearing simple clothes. "
    "Rich detailed painted backgrounds with depth and atmospheric lighting. "
    "Smooth animation, warm earthy color palette, web comic art style. "
    "High quality digital animation with painterly textured backgrounds."
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def emit_progress(session_id, step, progress, message, data=None):
    """Emit progress updates via WebSocket."""
    payload = {
        'session_id': session_id,
        'step': step,
        'progress': progress,
        'message': message
    }
    if data:
        payload['data'] = data
    socketio.emit('progress', payload)
    logger.info(f"[{session_id}] {step}: {message} ({progress}%)")


def transcribe_audio(filepath, session_id):
    """Transcribe audio using OpenAI Whisper API with timestamps."""
    emit_progress(session_id, 'transcription', 10, 'Starting transcription with Whisper...')

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    with open(filepath, 'rb') as audio_file:
        # Use verbose_json to get word-level timestamps
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    emit_progress(session_id, 'transcription', 50, 'Transcription complete, processing segments...')

    segments = []
    if hasattr(transcript, 'segments') and transcript.segments:
        for seg in transcript.segments:
            # Handle both object attributes and dict-style access
            if hasattr(seg, 'start'):
                start = seg.start
                end = seg.end
                text = seg.text
            else:
                start = seg['start']
                end = seg['end']
                text = seg['text']
            segments.append({
                'start': start,
                'end': end,
                'text': text.strip()
            })

    full_text = transcript.text if hasattr(transcript, 'text') else str(transcript)

    emit_progress(session_id, 'transcription', 100, f'Transcribed {len(segments)} segments')

    return {
        'full_text': full_text,
        'segments': segments
    }


def detect_scene_changes(transcript_data, session_id):
    """Use GPT-4 to analyze transcript and identify scene/topic changes with visual descriptions."""
    emit_progress(session_id, 'scene_detection', 10, 'Analyzing script for scene changes...')

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    segments_text = "\n".join([
        f"[{s['start']:.1f}s - {s['end']:.1f}s]: {s['text']}"
        for s in transcript_data['segments']
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a visual director for an animated story. Analyze the voiceover transcript "
                    "and identify distinct scenes/topic changes. For each scene, provide:\n"
                    "1. The start and end timestamps\n"
                    "2. A visual description for a stick figure paint-style illustration\n"
                    "3. Whether it's in the first 30 seconds (for video animation) or after (for still image)\n\n"
                    "The visual descriptions should be specific, depicting stick figures in action "
                    "that match what's being narrated. Keep descriptions consistent with a charming "
                    "hand-drawn paint/watercolor style.\n\n"
                    "Return valid JSON only, no markdown, with this structure:\n"
                    "{\n"
                    '  "scenes": [\n'
                    "    {\n"
                    '      "scene_number": 1,\n'
                    '      "start_time": 0.0,\n'
                    '      "end_time": 8.5,\n'
                    '      "narration_summary": "brief summary of what is being said",\n'
                    '      "visual_description": "specific description of stick figure scene to illustrate",\n'
                    '      "is_video": true\n'
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "Important rules:\n"
                    "- Scenes in the first 30 seconds should have is_video: true\n"
                    "- Scenes after 30 seconds should have is_video: false\n"
                    "- Each scene should be 3-10 seconds long\n"
                    "- Video scenes (first 30s) can be grouped into 2-4 scenes\n"
                    "- Cover the entire duration of the audio, no gaps\n"
                    "- Visual descriptions should tell a visual story matching the narration"
                )
            },
            {
                "role": "user",
                "content": f"Here is the transcribed voiceover with timestamps:\n\n{segments_text}"
            }
        ],
        temperature=0.7,
        max_tokens=4000
    )

    response_text = response.choices[0].message.content.strip()
    # Clean potential markdown wrapping
    if response_text.startswith('```'):
        response_text = response_text.split('\n', 1)[1]
        if response_text.endswith('```'):
            response_text = response_text[:-3]

    scenes = json.loads(response_text)

    emit_progress(session_id, 'scene_detection', 100,
                  f'Detected {len(scenes["scenes"])} scenes')

    return scenes['scenes']


def generate_image_imagen(prompt, output_path, session_id, scene_num):
    """Generate a still image using Gemini's native image generation."""
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        full_prompt = f"{STYLE_PROMPT} Scene: {prompt}"
        
        logger.info(f"Image generation request for scene {scene_num}")

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Image', 'Text']
            )
        )

        # Extract image from response parts
        image_saved = False
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    with open(output_path, 'wb') as f:
                        f.write(part.inline_data.data)
                    logger.info(f"Generated image for scene {scene_num}: {output_path}")
                    image_saved = True
                    break

        if not image_saved:
            logger.warning(f"No image in Gemini response for scene {scene_num}, trying Imagen fallback...")
            # Try Imagen as fallback
            try:
                response2 = client.models.generate_images(
                    model='imagen-3.0-generate-002',
                    prompt=full_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio='16:9',
                        safety_filter_level='BLOCK_ONLY_HIGH',
                    )
                )
                if response2.generated_images and len(response2.generated_images) > 0:
                    img_bytes = response2.generated_images[0].image.image_bytes
                    with open(output_path, 'wb') as f:
                        f.write(img_bytes)
                    logger.info(f"Imagen fallback succeeded for scene {scene_num}")
                    image_saved = True
            except Exception as e2:
                logger.warning(f"Imagen fallback also failed: {e2}")

        if not image_saved:
            logger.warning(f"All image generation failed for scene {scene_num}, using placeholder")
            emit_progress(session_id, 'generation', 0,
                         f'Image generation failed for scene {scene_num} — using placeholder')
            create_placeholder_image(prompt, output_path)

        return True

    except Exception as e:
        logger.error(f"Image generation failed for scene {scene_num}: {type(e).__name__}: {e}", exc_info=True)
        emit_progress(session_id, 'generation', 0,
                     f'Image error for scene {scene_num}: {str(e)[:100]}')
        create_placeholder_image(prompt, output_path)
        return True


def generate_video_veo(prompt, output_path, duration_seconds, session_id, scene_num):
    """Generate animated video using Google Veo API."""
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        full_prompt = f"{VIDEO_STYLE_PROMPT} Action: {prompt}"

        # Generate video with Veo
        operation = client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=full_prompt,
            config=types.GenerateVideosConfig(
                person_generation="allow_all",
                aspect_ratio="16:9",
                number_of_videos=1,
            )
        )

        # Poll for completion
        max_wait = 300  # 5 minutes max
        waited = 0
        while not operation.done and waited < max_wait:
            time.sleep(10)
            waited += 10
            operation = client.operations.get(operation)
            emit_progress(session_id, 'video_generation', 
                         min(30 + (waited / max_wait * 40), 70),
                         f'Generating video for scene {scene_num}... ({waited}s)')

        if operation.done and operation.response and operation.response.generated_videos:
            video = operation.response.generated_videos[0]
            # Download the video
            video_data = client.files.download(file=video.video)
            
            # video_data is a generator of bytes, collect all chunks
            video_bytes = b""
            for chunk in video_data:
                video_bytes += chunk
            
            with open(output_path, 'wb') as f:
                f.write(video_bytes)
            
            logger.info(f"Generated video for scene {scene_num}: {output_path}")
            return True
        else:
            logger.warning(f"Veo generation incomplete for scene {scene_num}, using image fallback")
            # Fallback to still image
            img_path = output_path.replace('.mp4', '.png')
            generate_image_imagen(prompt, img_path, session_id, scene_num)
            create_video_from_image(img_path, output_path, duration_seconds)
            return True

    except Exception as e:
        logger.error(f"Veo generation failed for scene {scene_num}: {e}")
        # Fallback: generate still image and convert to video
        img_path = output_path.replace('.mp4', '_fallback.png')
        generate_image_imagen(prompt, img_path, session_id, scene_num)
        create_video_from_image(img_path, output_path, duration_seconds)
        return True


def create_placeholder_image(prompt, output_path):
    """Create a simple placeholder image with text."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (1920, 1080), color=(255, 253, 245))
    draw = ImageDraw.Draw(img)

    # Draw a simple stick figure
    cx, cy = 960, 400
    # Head
    draw.ellipse([cx-40, cy-140, cx+40, cy-60], outline='#333333', width=3)
    # Body
    draw.line([cx, cy-60, cx, cy+60], fill='#333333', width=3)
    # Arms
    draw.line([cx-60, cy-20, cx+60, cy-20], fill='#333333', width=3)
    # Legs
    draw.line([cx, cy+60, cx-40, cy+140], fill='#333333', width=3)
    draw.line([cx, cy+60, cx+40, cy+140], fill='#333333', width=3)

    # Add scene text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    # Wrap text
    words = prompt[:200].split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) < 60:
            current_line = (current_line + " " + word).strip()
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    y_text = 600
    for line in lines[:4]:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((1920 - w) / 2, y_text), line, fill='#666666', font=font)
        y_text += 35

    img.save(output_path, 'PNG')


def create_video_from_image(image_path, video_path, duration):
    """Convert a still image to a video using ffmpeg with low memory usage."""
    try:
        # First resize the image to 1280x720 to reduce memory
        resized_path = image_path.replace('.png', '_resized.png')
        resize_cmd = [
            'ffmpeg', '-y',
            '-i', image_path,
            '-vf', 'scale=1280:720',
            resized_path
        ]
        subprocess.run(resize_cmd, check=True, capture_output=True, timeout=60)
        
        # Simple static image to video — minimal memory usage
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', resized_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'stillimage',
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=1280:720',
            '-r', '15',
            '-threads', '1',
            video_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        
        # Clean up resized image
        if os.path.exists(resized_path):
            os.remove(resized_path)
    except Exception as e:
        logger.error(f"Video from image failed: {e}, trying minimal approach")
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=640:360',
            '-r', '10',
            '-threads', '1',
            video_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)


def get_audio_duration(filepath):
    """Get audio duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def compose_final_video(scene_videos, audio_path, output_path, session_id):
    """Concatenate all scene videos and overlay the original audio."""
    emit_progress(session_id, 'compositing', 10, 'Compositing final video...')

    # Create concat file
    concat_path = os.path.join(os.path.dirname(output_path), 'concat.txt')
    with open(concat_path, 'w') as f:
        for vid in scene_videos:
            f.write(f"file '{vid}'\n")

    # Concatenate videos
    temp_video = os.path.join(os.path.dirname(output_path), 'temp_concat.mp4')
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', concat_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-r', '25',
        temp_video
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)

    emit_progress(session_id, 'compositing', 60, 'Adding audio track...')

    # Add audio
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)

    # Cleanup
    os.remove(concat_path)
    os.remove(temp_video)

    emit_progress(session_id, 'compositing', 100, 'Final video complete!')

    return output_path


def process_voiceover(filepath, session_id):
    """Main pipeline: transcribe -> detect scenes -> generate visuals -> compose video."""
    try:
        work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Get audio duration
        audio_duration = get_audio_duration(filepath)
        emit_progress(session_id, 'init', 5, f'Audio duration: {audio_duration:.1f}s')

        # Step 2: Transcribe
        transcript_data = transcribe_audio(filepath, session_id)

        # Step 3: Detect scene changes
        scenes = detect_scene_changes(transcript_data, session_id)

        # Save scene data
        scene_data_path = os.path.join(work_dir, 'scenes.json')
        with open(scene_data_path, 'w') as f:
            json.dump({
                'transcript': transcript_data,
                'scenes': scenes,
                'audio_duration': audio_duration
            }, f, indent=2)

        # Step 4: Generate visuals for each scene
        scene_videos = []
        total_scenes = len(scenes)

        for i, scene in enumerate(scenes):
            scene_num = scene.get('scene_number', i + 1)
            start = scene['start_time']
            end = scene['end_time']
            duration = end - start
            is_video = scene.get('is_video', start < 30)
            visual_desc = scene['visual_description']

            progress_base = 20 + (60 * i / total_scenes)
            emit_progress(session_id, 'generation', int(progress_base),
                         f'Generating visual for scene {scene_num}/{total_scenes}...')

            if is_video:
                # Generate animated video with Veo for first 30 seconds
                video_path = os.path.join(work_dir, f'scene_{scene_num:03d}.mp4')
                generate_video_veo(visual_desc, video_path, duration, session_id, scene_num)

                # Trim to exact duration needed
                trimmed_path = os.path.join(work_dir, f'scene_{scene_num:03d}_trimmed.mp4')
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-r', '25',
                    '-vf', 'scale=1920:1080',
                    trimmed_path
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
                    scene_videos.append(trimmed_path)
                except Exception:
                    scene_videos.append(video_path)
            else:
                # Generate still image with Imagen
                img_path = os.path.join(work_dir, f'scene_{scene_num:03d}.png')
                generate_image_imagen(visual_desc, img_path, session_id, scene_num)

                # Convert to video segment with Ken Burns effect
                video_path = os.path.join(work_dir, f'scene_{scene_num:03d}_video.mp4')
                create_video_from_image(img_path, video_path, duration)
                scene_videos.append(video_path)

            emit_progress(session_id, 'generation', int(progress_base + (60 / total_scenes)),
                         f'Scene {scene_num}/{total_scenes} complete')

        # Step 5: Compose final video
        output_filename = f'visualized_{session_id}.mp4'
        output_path = os.path.join(work_dir, output_filename)
        compose_final_video(scene_videos, filepath, output_path, session_id)

        emit_progress(session_id, 'complete', 100, 'Processing complete!', {
            'video_url': f'/download/{session_id}/{output_filename}',
            'scenes': scenes
        })

        return output_path

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        emit_progress(session_id, 'error', 0, f'Error: {str(e)}')
        raise


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    session_id = str(uuid.uuid4())[:12]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_{filename}')
    file.save(filepath)

    # Start processing in background
    socketio.start_background_task(process_voiceover, filepath, session_id)

    return jsonify({
        'session_id': session_id,
        'message': 'Processing started',
        'filename': filename
    })


@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    work_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    return send_from_directory(work_dir, filename, as_attachment=True)


@app.route('/scenes/<session_id>')
def get_scenes(session_id):
    scene_file = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'scenes.json')
    if os.path.exists(scene_file):
        with open(scene_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Scenes not found'}), 404


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'openai_configured': bool(OPENAI_API_KEY),
        'google_configured': bool(GOOGLE_API_KEY)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
