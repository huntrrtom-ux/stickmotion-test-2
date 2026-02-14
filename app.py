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

# =============================================================
# STYLE SEED — condensed for consistency without overwhelming the scene
# =============================================================
STYLE_SHORT = (
    "Art style: cartoon stick figures with oversized round white heads (thick black outline), "
    "expressive black eyebrows, small dot eyes, line mouth, no nose/ears. "
    "Bodies wear muted clothing (khaki, olive, brown) with visible folds. "
    "White mitten hands, dark shoes. Subtle chin shadow and ground shadow. "
    "Backgrounds: richly painted, layered, warm earthy tones, atmospheric golden lighting, "
    "like an animated adventure film. Painterly textured environments. "
    "16:9 cinematic. No text/words/watermarks in image."
)

# Full style seed kept for reference/Veo
STYLE_SEED = STYLE_SHORT

STYLE_PROMPT = STYLE_SHORT

VIDEO_STYLE_PROMPT = (
    f"{STYLE_SHORT} Animated with smooth subtle movement."
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
                    "You are a visual director for an animated story using CONSISTENT stick figure characters. "
                    "Analyze the voiceover transcript and identify distinct scenes/topic changes.\n\n"
                    "CHARACTER DESIGN (IDENTICAL in every scene — never deviate):\n"
                    "- Every character has an oversized round white head with thick black outline\n"
                    "- Thick expressive black eyebrows, small black oval eyes, simple line mouth\n"
                    "- NO nose, NO ears, NO hair (unless wearing a hat)\n"
                    "- Bodies wear muted natural-colored clothing: khaki, olive, brown, dark gray\n"
                    "- Small white mitten hands, simple dark shoes\n"
                    "- Characters are distinguished ONLY by clothing color, hats, and accessories\n"
                    "- Do NOT describe heads/faces differently between scenes — only describe clothing and actions\n\n"
                    "For each scene provide:\n"
                    "1. The start and end timestamps\n"
                    "2. A visual description focusing on CHARACTER ACTIONS and DETAILED BACKGROUND/SETTING\n"
                    "3. Whether it's in the first 30 seconds (for video animation) or after (for still image)\n\n"
                    "CRITICAL RULES FOR VISUAL DESCRIPTIONS:\n"
                    "- Do NOT describe the character's head/face style (the image generator already knows this)\n"
                    "- Instead focus on: what the character WEARS, what they're DOING, and the SETTING\n"
                    "- Describe rich backgrounds: environment type, lighting direction, color mood, foreground/background elements\n"
                    "- Example: 'Character in olive jacket points ahead, standing on a misty forest path. Golden light filters through dense canopy. Hanging vines and moss-covered rocks frame the scene.'\n"
                    "- Limit to 1-3 characters per scene for clarity\n"
                    "- Do NOT include any text, words, labels, or watermarks in the visual description\n\n"
                    "Return valid JSON only, no markdown, with this structure:\n"
                    "{\n"
                    '  "scenes": [\n'
                    "    {\n"
                    '      "scene_number": 1,\n'
                    '      "start_time": 0.0,\n'
                    '      "end_time": 8.5,\n'
                    '      "narration_summary": "brief summary of what is being said",\n'
                    '      "visual_description": "A stick figure with a round white head and dot eyes, wearing a blue shirt, stands in [detailed scene description]",\n'
                    '      "is_video": true\n'
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "Important rules:\n"
                    "- Scenes in the first 30 seconds should have is_video: true\n"
                    "- Scenes after 30 seconds should have is_video: false\n"
                    "- GENERATE A NEW SCENE FOR EVERY SENTENCE OR DISTINCT BEAT in the narration\n"
                    "- Each scene should be 2-5 seconds long — SHORT and frequent\n"
                    "- Aim for at least one scene every 3-5 seconds of audio\n"
                    "- A 1-minute voiceover should have 12-20 scenes\n"
                    "- A 3-minute voiceover should have 35-60 scenes\n"
                    "- Every new idea, action, or sentence deserves its own visual\n"
                    "- Cover the entire duration of the audio, no gaps\n"
                    "- Visual descriptions should tell a visual story matching the narration\n"
                    "- EVERY visual_description must start with the character description for consistency"
                )
            },
            {
                "role": "user",
                "content": f"Here is the transcribed voiceover with timestamps:\n\n{segments_text}"
            }
        ],
        temperature=0.7,
        max_tokens=12000
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


def get_whisk_token(session_id):
    """Get a bearer token from the Whisk cookie. Caches token for reuse."""
    import requests as req
    
    # Check if we have a cached token that's still valid
    if hasattr(get_whisk_token, '_token') and hasattr(get_whisk_token, '_expiry'):
        if time.time() < get_whisk_token._expiry:
            return get_whisk_token._token
    
    WHISK_COOKIE = os.environ.get('WHISK_COOKIE', '')
    WHISK_API_KEY = os.environ.get('WHISK_API_KEY', '')
    
    # If WHISK_API_KEY is set directly (bearer token), use it
    if WHISK_API_KEY and not WHISK_COOKIE:
        get_whisk_token._token = WHISK_API_KEY
        get_whisk_token._expiry = time.time() + 3600  # assume 1 hour
        return WHISK_API_KEY
    
    # Otherwise, exchange cookie for token
    if WHISK_COOKIE:
        try:
            resp = req.get(
                "https://labs.google/fx/api/auth/session",
                headers={"cookie": WHISK_COOKIE},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                if "access_token" in data:
                    get_whisk_token._token = data["access_token"]
                    # Parse expiry or default to 1 hour
                    get_whisk_token._expiry = time.time() + 3500
                    logger.info("Whisk token refreshed successfully")
                    return data["access_token"]
                else:
                    logger.error(f"No access_token in session response: {data}")
            else:
                logger.error(f"Session refresh failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
    
    # Fallback to WHISK_API_KEY
    return WHISK_API_KEY


def generate_image_whisk(prompt, output_path, session_id, scene_num):
    """Generate a still image using Google Whisk (unofficial API). Whisk only — no fallback."""
    import requests as req
    
    token = get_whisk_token(session_id)
    
    full_prompt = f"{STYLE_PROMPT}\n\nSCENE: {prompt}"
    
    workflow_id = str(uuid.uuid4())
    session_ts = f";{int(time.time() * 1000)}"
    
    # Exact payload from browser Network tab
    json_data = {
        "clientContext": {
            "workflowId": workflow_id,
            "tool": "BACKBONE",
            "sessionId": session_ts
        },
        "imageModelSettings": {
            "imageModel": "IMAGEN_3_5",
            "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"
        },
        "mediaCategory": "MEDIA_CATEGORY_BOARD",
        "prompt": full_prompt,
        "seed": 0
    }
    
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "origin": "https://labs.google",
        "referer": "https://labs.google/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }
    
    logger.info(f"Whisk image generation for scene {scene_num}")
    
    response = req.post(
        "https://aisandbox-pa.googleapis.com/v1/whisk:generateImage",
        json=json_data,
        headers=headers,
        timeout=120
    )
    
    logger.info(f"Whisk response status for scene {scene_num}: {response.status_code}")
    
    if response.status_code != 200:
        error_msg = f"Whisk API error {response.status_code}: {response.text[:300]}"
        logger.error(error_msg)
        emit_progress(session_id, 'generation', 0, error_msg)
        create_placeholder_image(prompt, output_path)
        return None
    
    result = response.json()
    
    if "imagePanels" in result and result["imagePanels"]:
        image_panel = result["imagePanels"][0]
        if "generatedImages" in image_panel and image_panel["generatedImages"]:
            img_data = image_panel["generatedImages"][0]
            encoded_image = img_data["encodedImage"]
            
            # Remove data URL prefix if present
            if "," in encoded_image:
                encoded_image = encoded_image.split(",", 1)[1]
            
            image_bytes = base64.b64decode(encoded_image)
            
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            
            # Return media info for potential animation
            media_id = img_data.get("mediaGenerationId", None)
            img_prompt = img_data.get("prompt", prompt)
            encoded_for_animate = img_data["encodedImage"]  # keep original for animate
            logger.info(f"Whisk generated image for scene {scene_num}: {output_path} (media_id: {media_id})")
            return {
                "media_id": media_id,
                "prompt": img_prompt,
                "encoded_image": encoded_for_animate,
                "workflow_id": img_data.get("workflowId", "")
            }
    
    logger.warning(f"Whisk returned no images for scene {scene_num}")
    emit_progress(session_id, 'generation', 0, f'Whisk returned no images for scene {scene_num}')
    create_placeholder_image(prompt, output_path)
    return None


def animate_image_whisk(image_info, script, output_path, session_id, scene_num):
    """Animate a Whisk-generated image into video using Whisk Animate (Veo)."""
    import requests as req
    
    token = get_whisk_token(session_id)
    
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "origin": "https://labs.google",
        "referer": "https://labs.google/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }
    
    # Step 1: Start the video generation (exact payload from browser Network tab)
    session_ts = f";{int(time.time() * 1000)}"
    
    # Build prompt with "ORIGINAL IMAGE DESCRIPTION:" prefix (matches browser)
    original_prompt = image_info.get("prompt", script)
    prefixed_prompt = f"ORIGINAL IMAGE DESCRIPTION:\n{original_prompt}"
    
    # Ensure rawBytes is clean base64 (strip data URL prefix if present)
    raw_bytes_for_anim = image_info.get("encoded_image", "")
    if raw_bytes_for_anim and "," in raw_bytes_for_anim[:100]:
        raw_bytes_for_anim = raw_bytes_for_anim.split(",", 1)[1]
    
    animate_data = {
        "clientContext": {
            "sessionId": session_ts,
            "tool": "BACKBONE",
            "workflowId": image_info.get("workflow_id", str(uuid.uuid4()))
        },
        "loopVideo": False,
        "modelKey": "",
        "modelNameType": "VEO_3_1_I2V_12STEP",
        "promptImageInput": {
            "mediaGenerationId": image_info.get("media_id", ""),
            "prompt": prefixed_prompt,
            "rawBytes": raw_bytes_for_anim,
        },
        "userInstructions": "",
    }
    
    logger.info(f"Whisk Animate starting for scene {scene_num}, rawBytes length: {len(animate_data['promptImageInput']['rawBytes'])}, mediaGenerationId: {animate_data['promptImageInput']['mediaGenerationId'][:50] if animate_data['promptImageInput']['mediaGenerationId'] else 'NONE'}")
    
    response = req.post(
        "https://aisandbox-pa.googleapis.com/v1/whisk:generateVideo",
        json=animate_data,
        headers=headers,
        timeout=60
    )
    
    logger.info(f"Whisk Animate response for scene {scene_num}: status={response.status_code}")
    
    if response.status_code != 200:
        logger.error(f"Whisk Animate failed for scene {scene_num}: {response.status_code} - {response.text[:500]}")
        return False
    
    result = response.json()
    logger.info(f"Whisk Animate result keys for scene {scene_num}: {list(result.keys())}")
    logger.info(f"Whisk Animate full result for scene {scene_num}: {json.dumps(result)[:500]}")
    
    # Extract operation name for polling
    operation_name = None
    if "operation" in result:
        op = result["operation"]
        logger.info(f"operation keys: {list(op.keys()) if isinstance(op, dict) else type(op)}")
        if isinstance(op, dict) and "operation" in op:
            operation_name = op["operation"].get("name", "")
        elif isinstance(op, dict) and "name" in op:
            operation_name = op.get("name", "")
    if not operation_name and "name" in result:
        operation_name = result["name"]
    
    if not operation_name:
        logger.error(f"No operation name returned for scene {scene_num}: {json.dumps(result)[:300]}")
        return False
    
    logger.info(f"Whisk Animate operation: {operation_name}")
    
    # Step 2: Poll for completion using the correct status check endpoint
    max_polls = 90  # ~180 seconds (3 min) max — video gen takes time
    
    for i in range(max_polls):
        time.sleep(2)
        
        emit_progress(session_id, 'generation', 0,
                     f'Animating scene {scene_num}... ({(i+1)*2}s)')
        
        poll_data = {
            "operations": [{"operation": {"name": operation_name}}]
        }
        
        poll_response = req.post(
            "https://aisandbox-pa.googleapis.com/v1:runVideoFxSingleClipsStatusCheck",
            json=poll_data,
            headers=headers,
            timeout=30
        )
        
        if poll_response.status_code != 200:
            logger.warning(f"Poll failed for scene {scene_num}: {poll_response.status_code}")
            continue
        
        poll_result = poll_response.json()
        status = poll_result.get("status", "")
        
        # Log all keys on first poll and when status changes
        if i == 0:
            logger.info(f"Poll result keys for scene {scene_num}: {list(poll_result.keys())}")
        
        logger.info(f"Animate poll {i+1} for scene {scene_num}: status={status}")
        
        if status == "MEDIA_GENERATION_STATUS_SUCCESSFUL":
            # Log the full response structure (truncate rawBytes)
            debug_keys = {k: (f"len={len(v)}" if k == "rawBytes" and v else str(v)[:200]) for k, v in poll_result.items()}
            logger.info(f"SUCCESS response for scene {scene_num}: {json.dumps(debug_keys)}")
            
            # Extract video bytes
            raw_bytes = poll_result.get("rawBytes", "")
            if raw_bytes:
                if "," in raw_bytes:
                    raw_bytes = raw_bytes.split(",", 1)[1]
                
                video_bytes = base64.b64decode(raw_bytes)
                with open(output_path, 'wb') as f:
                    f.write(video_bytes)
                
                logger.info(f"Whisk Animate generated video for scene {scene_num}: {output_path}")
                return True
            
            logger.warning(f"Animation successful but no rawBytes for scene {scene_num}")
            return False
        
        if status == "MEDIA_GENERATION_STATUS_FAILED":
            logger.error(f"Animation failed for scene {scene_num}")
            return False
    
    logger.warning(f"Whisk Animate timed out for scene {scene_num}")
    return False


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
        animated_count = 0
        MAX_ANIMATED_SCENES = 999  # No limit

        for i, scene in enumerate(scenes):
            scene_num = scene.get('scene_number', i + 1)
            start = scene['start_time']
            end = scene['end_time']
            duration = end - start
            is_video = True  # Always animate every scene
            visual_desc = scene['visual_description']

            logger.info(f"Scene {scene_num}: start={start}, i={i}, is_video={is_video}")

            progress_base = 20 + (60 * i / total_scenes)
            emit_progress(session_id, 'generation', int(progress_base),
                         f'Generating visual for scene {scene_num}/{total_scenes}...')

            # Step 1: Always generate a Whisk image first
            img_path = os.path.join(work_dir, f'scene_{scene_num:03d}.png')
            image_info = generate_image_whisk(visual_desc, img_path, session_id, scene_num)
            
            logger.info(f"Scene {scene_num}: is_video={is_video}, image_info_type={type(image_info).__name__}, animated_count={animated_count}/{MAX_ANIMATED_SCENES}")

            if is_video and image_info and animated_count < MAX_ANIMATED_SCENES:
                # Step 2: Animate the image using Whisk Animate (Veo)
                logger.info(f"Scene {scene_num} is_video=True, attempting animation ({animated_count+1}/{MAX_ANIMATED_SCENES})...")
                video_path = os.path.join(work_dir, f'scene_{scene_num:03d}_animated.mp4')
                try:
                    animated = animate_image_whisk(
                        image_info, visual_desc, video_path, session_id, scene_num
                    )
                except Exception as e:
                    logger.error(f"Animation exception for scene {scene_num}: {e}", exc_info=True)
                    animated = False
                
                if animated:
                    animated_count += 1
                
                if animated:
                    # Trim to exact duration
                    trimmed_path = os.path.join(work_dir, f'scene_{scene_num:03d}_trimmed.mp4')
                    try:
                        cmd = [
                            'ffmpeg', '-y',
                            '-i', video_path,
                            '-t', str(duration),
                            '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            '-r', '25',
                            '-vf', 'scale=1280:720',
                            '-threads', '1',
                            trimmed_path
                        ]
                        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
                        scene_videos.append(trimmed_path)
                    except Exception:
                        scene_videos.append(video_path)
                else:
                    # Animation failed — use still image with Ken Burns
                    video_path = os.path.join(work_dir, f'scene_{scene_num:03d}_video.mp4')
                    create_video_from_image(img_path, video_path, duration)
                    scene_videos.append(video_path)
            else:
                # Still image scene — Ken Burns effect
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


@app.route('/test-animate')
def test_animate():
    """Test endpoint — generates 1 Whisk image and tries to animate it."""
    import requests as req
    
    results = {"steps": []}
    
    try:
        token = get_whisk_token("test")
        results["steps"].append({"step": "token", "ok": True, "prefix": token[:20] + "..."})
    except Exception as e:
        results["steps"].append({"step": "token", "ok": False, "error": str(e)})
        return jsonify(results)
    
    prompt = "A cartoon character with a big round white head standing in a sunny meadow"
    wf_id = str(uuid.uuid4())
    
    img_payload = {
        "clientContext": {"workflowId": wf_id, "tool": "BACKBONE", "sessionId": f";{int(time.time()*1000)}"},
        "imageModelSettings": {"imageModel": "IMAGEN_3_5", "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE"},
        "mediaCategory": "MEDIA_CATEGORY_BOARD",
        "prompt": prompt,
        "seed": 0
    }
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "origin": "https://labs.google",
        "referer": "https://labs.google/",
        "user-agent": "Mozilla/5.0"
    }
    
    try:
        r = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:generateImage", json=img_payload, headers=headers, timeout=60)
        if r.status_code != 200:
            results["steps"].append({"step": "image", "status": r.status_code, "error": r.text[:300]})
            return jsonify(results)
        
        img_result = r.json()
        encoded_image = None
        media_gen_id = ""
        for panel in img_result.get("imagePanels", []):
            for img in panel.get("generatedImages", []):
                encoded_image = img.get("encodedImage", "")
                media_gen_id = img.get("mediaGenerationId", "")
                break
        
        results["steps"].append({"step": "image", "ok": True, "img_len": len(encoded_image or ""), "media_id": media_gen_id[:50] if media_gen_id else "NONE"})
        
        if not encoded_image:
            results["steps"].append({"step": "error", "msg": "no encoded image", "keys": str(img_result.keys())})
            return jsonify(results)
    except Exception as e:
        results["steps"].append({"step": "image", "ok": False, "error": str(e)})
        return jsonify(results)
    
    # Try animate (exact payload from browser Network tab)
    anim_payload = {
        "clientContext": {
            "sessionId": sess_id,
            "tool": "BACKBONE",
            "workflowId": wf_id
        },
        "loopVideo": False,
        "modelKey": "",
        "modelNameType": "VEO_3_1_I2V_12STEP",
        "promptImageInput": {
            "mediaGenerationId": media_gen_id,
            "prompt": f"ORIGINAL IMAGE DESCRIPTION:\n{prompt}",
            "rawBytes": encoded_image
        },
        "userInstructions": "",
    }
    
    try:
        ar = req.post("https://aisandbox-pa.googleapis.com/v1/whisk:generateVideo", json=anim_payload, headers=headers, timeout=60)
        if ar.status_code != 200:
            results["steps"].append({"step": "animate", "status": ar.status_code, "error": ar.text[:500]})
            return jsonify(results)
        
        anim_result = ar.json()
        op_name = None
        if "operation" in anim_result and "operation" in anim_result["operation"]:
            op_name = anim_result["operation"]["operation"].get("name", "")
        elif "name" in anim_result:
            op_name = anim_result["name"]
        
        results["steps"].append({"step": "animate", "ok": True, "operation": op_name, "resp_keys": list(anim_result.keys())})
        
        if not op_name:
            results["steps"].append({"step": "animate_detail", "full_response": str(anim_result)[:500]})
            return jsonify(results)
        
        # Poll until done (max 60 polls = 2 min)
        for poll_i in range(60):
            time.sleep(2)
            poll = req.post("https://aisandbox-pa.googleapis.com/v1:runVideoFxSingleClipsStatusCheck",
                           json={"operations": [{"operation": {"name": op_name}}]}, headers=headers, timeout=30)
            if poll.status_code == 200:
                pb = poll.json()
                status = pb.get("status", "?")
                if status == "MEDIA_GENERATION_STATUS_SUCCESSFUL":
                    # Show full structure (but truncate rawBytes)
                    debug_pb = {}
                    for k, v in pb.items():
                        if k == "rawBytes":
                            debug_pb[k] = f"(length={len(v)})" if v else "EMPTY"
                        else:
                            debug_pb[k] = str(v)[:200]
                    results["steps"].append({"step": "poll_done", "poll_num": poll_i+1, "status": status, "response_structure": debug_pb, "all_keys": list(pb.keys())})
                    return jsonify(results)
                elif status == "MEDIA_GENERATION_STATUS_FAILED":
                    results["steps"].append({"step": "poll_failed", "poll_num": poll_i+1, "full_response": str(pb)[:500]})
                    return jsonify(results)
                # Still active, continue polling
            else:
                results["steps"].append({"step": "poll_error", "poll_num": poll_i+1, "status_code": poll.status_code, "error": poll.text[:300]})
                return jsonify(results)
        
        results["steps"].append({"step": "poll_timeout", "message": "Still generating after 2 minutes"})
    except Exception as e:
        results["steps"].append({"step": "animate", "ok": False, "error": str(e)})
    
    return jsonify(results)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
