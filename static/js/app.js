// ============================================
// STICKMOTION — Frontend Application
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const fileSelected = document.getElementById('file-selected');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const generateBtn = document.getElementById('generate-btn');
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultSection = document.getElementById('result-section');
    const statusMessage = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const sceneTimeline = document.getElementById('scene-timeline');
    const downloadBtn = document.getElementById('download-btn');
    const newBtn = document.getElementById('new-btn');

    let selectedFile = null;
    let currentSessionId = null;

    // WebSocket
    const socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('progress', (data) => {
        if (data.session_id !== currentSessionId) return;
        handleProgress(data);
    });

    // ---- File Selection ----

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectFile(e.target.files[0]);
        }
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            selectFile(e.dataTransfer.files[0]);
        }
    });

    function selectFile(file) {
        const allowedTypes = ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'webm', 'mp4'];
        const ext = file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(ext)) {
            alert(`Invalid file type ".${ext}". Please upload: ${allowedTypes.join(', ')}`);
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileSelected.classList.remove('hidden');
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // ---- Generate ----

    generateBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        generateBtn.disabled = true;
        generateBtn.querySelector('.btn-text').textContent = 'Uploading...';

        const formData = new FormData();
        formData.append('audio', selectedFile);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                currentSessionId = data.session_id;
                showProcessing();
            } else {
                alert(data.error || 'Upload failed');
                resetUploadBtn();
            }
        } catch (err) {
            console.error('Upload error:', err);
            alert('Failed to upload file. Please try again.');
            resetUploadBtn();
        }
    });

    function resetUploadBtn() {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = 'Generate Visuals';
    }

    // ---- Processing ----

    function showProcessing() {
        uploadSection.classList.add('hidden');
        processingSection.classList.remove('hidden');
        resultSection.classList.add('hidden');

        // Reset steps
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active', 'complete');
        });
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
    }

    function handleProgress(data) {
        const { step, progress, message } = data;

        // Update status message
        statusMessage.textContent = message;

        // Update progress bar
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;

        // Update step indicators
        const stepOrder = ['transcription', 'scene_detection', 'generation', 'compositing'];
        const currentIndex = stepOrder.indexOf(step);

        stepOrder.forEach((s, i) => {
            const el = document.getElementById(`step-${s}`);
            if (!el) return;

            if (i < currentIndex) {
                el.classList.remove('active');
                el.classList.add('complete');
            } else if (i === currentIndex) {
                el.classList.add('active');
                el.classList.remove('complete');
                if (progress >= 100 && step !== 'complete') {
                    el.classList.remove('active');
                    el.classList.add('complete');
                }
            } else {
                el.classList.remove('active', 'complete');
            }
        });

        // Handle completion
        if (step === 'complete' && data.data) {
            // Mark all steps complete
            stepOrder.forEach(s => {
                const el = document.getElementById(`step-${s}`);
                if (el) {
                    el.classList.remove('active');
                    el.classList.add('complete');
                }
            });

            setTimeout(() => {
                showResult(data.data);
            }, 800);
        }

        // Handle error
        if (step === 'error') {
            statusMessage.textContent = message;
            statusMessage.style.color = '#DC2626';
        }
    }

    // ---- Result ----

    function showResult(data) {
        processingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');

        // Set download link
        downloadBtn.href = data.video_url;

        // Build scene timeline
        sceneTimeline.innerHTML = '';

        if (data.scenes && data.scenes.length > 0) {
            data.scenes.forEach(scene => {
                const isVideo = scene.is_video;
                const item = document.createElement('div');
                item.className = 'scene-item';
                item.innerHTML = `
                    <div class="scene-badge ${isVideo ? 'video' : 'image'}">
                        ${scene.scene_number}
                    </div>
                    <div class="scene-details">
                        <div class="scene-time">
                            ${formatTime(scene.start_time)} — ${formatTime(scene.end_time)}
                        </div>
                        <div class="scene-desc">${escapeHtml(scene.visual_description)}</div>
                    </div>
                    <span class="scene-type-tag ${isVideo ? 'video' : 'image'}">
                        ${isVideo ? 'Veo' : 'Imagen'}
                    </span>
                `;
                sceneTimeline.appendChild(item);
            });
        }
    }

    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ---- New Upload ----

    newBtn.addEventListener('click', () => {
        selectedFile = null;
        currentSessionId = null;
        fileInput.value = '';
        fileSelected.classList.add('hidden');
        resetUploadBtn();
        statusMessage.style.color = '';

        resultSection.classList.add('hidden');
        processingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    });
});
