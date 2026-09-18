const recordBtn = document.getElementById('record-btn');
const statusIndicator = document.getElementById('status-indicator');
const transcriptBox = document.getElementById('transcript-box');
const intentBox = document.getElementById('intent-box');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceFill = document.querySelector('.confidence-fill');
const confidenceText = document.querySelector('.confidence-text');

let mediaRecorder = null;
let audioChunks = [];

async function setupAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];
            await sendAudioToAPI(audioBlob);
        };
    } catch (err) {
        console.error('Error accessing microphone:', err);
        statusIndicator.textContent = 'Microphone access denied';
        statusIndicator.classList.add('error');
    }
}

async function sendAudioToAPI(blob) {
    statusIndicator.textContent = 'Processing...';
    statusIndicator.classList.remove('error');
    
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');
    formData.append('schema_name', 'financial');

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'API request failed');
        }

        const data = await response.json();
        
        // Update UI
        transcriptBox.textContent = data.transcript;
        intentBox.textContent = JSON.stringify(data.intent, null, 2);
        
        // Update confidence
        const confPercent = Math.round(data.confidence * 100);
        confidenceBar.style.display = 'flex';
        confidenceFill.style.setProperty('--confidence-width', `${confPercent}%`);
        confidenceText.textContent = `${confPercent}%`;
        
        if (confPercent > 80) {
            confidenceFill.classList.add('high');
        } else {
            confidenceFill.classList.remove('high');
        }

        statusIndicator.textContent = 'Done';
    } catch (err) {
        console.error('API Error:', err);
        statusIndicator.textContent = err.message || 'Error processing audio';
        statusIndicator.classList.add('error');
        transcriptBox.textContent = 'Error processing audio.';
        intentBox.textContent = '{}';
    }
}

// Event Listeners for hold-to-record
recordBtn.addEventListener('mousedown', startRecording);
recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });

window.addEventListener('mouseup', stopRecording);
window.addEventListener('touchend', stopRecording);

function startRecording() {
    if (!mediaRecorder) return;
    
    // Reset UI
    transcriptBox.textContent = 'Listening...';
    intentBox.textContent = '{}';
    confidenceBar.style.display = 'none';
    
    audioChunks = [];
    mediaRecorder.start();
    
    recordBtn.classList.add('recording');
    recordBtn.querySelector('.btn-text').textContent = 'Recording...';
    statusIndicator.textContent = 'Listening';
    statusIndicator.classList.remove('error');
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.btn-text').textContent = 'Hold to Speak';
    }
}

// Init
setupAudio();
