import os
import gradio as gr
import torch
import cv2
import numpy as np
from collections import deque
from operator import itemgetter
import time
import threading
from datetime import datetime
import logging
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmaction.apis import init_recognizer
from mmaction.utils import get_str_type

# Configure logging
logging.getLogger("mmaction").setLevel(logging.WARNING)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000|max_delay;500000|reorder_queue_size;0|stimeout;1000000"

# Global variables for stream handling
stream_active = False
current_frame = None
detection_logs = []
last_notification_time = 0
NOTIFICATION_COOLDOWN = 5  # seconds

# Constants for visualization
FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

# Excluded steps for pipeline
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

def get_current_frame():
    """Get the current frame from the RTSP stream"""
    global current_frame
    return current_frame

def process_frame(frame):
    """Process a single frame with MMAction2"""
    global scores_sum, score_cache, frame_queue, data, model, test_pipeline, label, threshold, average_size
    if frame is None:
        return None
    
    # Add frame to queue
    frame_queue.append(np.array(frame[:, :, ::-1]))
    
    # Process if we have enough frames
    if len(frame_queue) == sample_length:
        cur_windows = list(np.array(frame_queue))
        if data['img_shape'] is None:
            data['img_shape'] = frame_queue.popleft().shape[:2]
        
        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])
        
        # Forward the model
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
        
        scores = result.pred_score.tolist()
        scores = np.array(scores)
        score_cache.append(scores)
        scores_sum += scores
        
        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)
            
            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]
            
            # Draw results on frame
            for i, (selected_label, score) in enumerate(results):
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = f"{selected_label}: {score:.2%}"
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                           FONTCOLOR, THICKNESS, LINETYPE)
                
                # Log detection
                detection_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {text}")
                if len(detection_logs) > 20:  # Keep only last 20 logs
                    detection_logs.pop(0)
            
            scores_sum -= score_cache.popleft()
    
    return frame

def stream_thread():
    """Background thread to handle RTSP stream"""
    global current_frame, stream_active
    
    cap = cv2.VideoCapture("rtsp://admin:akilicamera@154.70.45.143:554/mode=real&idc=1&ids=1")
    
    while stream_active:
        ret, frame = cap.read()
        if ret:
            current_frame = process_frame(frame)
        time.sleep(0.1)  # Small delay to prevent overwhelming the stream
    
    cap.release()

def start_stream():
    """Start the RTSP stream"""
    global stream_active
    stream_active = True
    threading.Thread(target=stream_thread, daemon=True).start()
    return "Stream started..."

def stop_stream():
    """Stop the RTSP stream"""
    global stream_active
    stream_active = False
    return "Stream stopped."

# Initialize MMAction2 model and variables
def init_model():
    global model, data, label, sample_length, test_pipeline, frame_queue, result_queue, \
           score_cache, scores_sum, average_size, threshold
    
    # Load config and checkpoint
    config = Config.fromfile('configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py')
    checkpoint = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
    
    # Initialize model
    model = init_recognizer(config, checkpoint, device='cpu')
    
    # Initialize data structures
    data = dict(img_shape=None, modality='RGB', label=-1)
    
    # Load labels
    with open('tools/data/kinetics/label_map_k400.txt', 'r') as f:
        label = [line.strip() for line in f]
    
    # Prepare test pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in get_str_type(step['type']):
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if get_str_type(step['type']) in EXCLUED_STEPS:
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)
    
    # Initialize queues and variables
    frame_queue = deque(maxlen=sample_length)
    result_queue = deque(maxlen=1)
    score_cache = deque()
    scores_sum = 0
    average_size = 1
    threshold = 0.01

# Create Gradio interface
with gr.Blocks(title="MMAction2 Action Recognition") as iface:
    gr.Markdown("# MMAction2 Action Recognition")
    
    with gr.Row():
        start_btn = gr.Button("Start Stream", variant="primary", scale=1)
        stop_btn = gr.Button("Stop Stream", variant="stop", scale=1)
        refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
    
    with gr.Row():
        with gr.Column(scale=2):
            camera_feed = gr.Image(
                label="Camera Feed",
                interactive=False,
                height=720
            )
        with gr.Column(scale=1):
            detection_log = gr.Textbox(
                label="Detection Logs",
                value="Ready to start...",
                lines=20,
                max_lines=20,
                interactive=False
            )
    
    def update_feed_and_logs():
        """Update both feed and logs"""
        frame = get_current_frame() if stream_active else None
        logs = "\n".join(detection_logs)
        return frame, logs
    
    # Update the stream handling
    start_btn.click(
        fn=start_stream,
        outputs=detection_log
    )
    
    stop_btn.click(
        fn=stop_stream,
        outputs=detection_log
    )
    
    # Add refresh button for updates
    refresh_btn.click(
        fn=update_feed_and_logs,
        outputs=[camera_feed, detection_log]
    )
    
    # Add auto-refresh using JavaScript
    gr.Markdown("""
    <script>
    function setupAutoRefresh() {
        function findRefreshButton() {
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                if (btn.textContent.includes('🔄')) {
                    return btn;
                }
            }
            return null;
        }
        
        let refreshInterval;
        
        function startAutoRefresh() {
            if (!refreshInterval) {
                const refreshBtn = findRefreshButton();
                if (refreshBtn) {
                    refreshInterval = setInterval(() => {
                        refreshBtn.click();
                    }, 200);
                }
            }
        }
        
        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
            }
        }
        
        const observer = new MutationObserver(() => {
            const startBtn = document.querySelector('button:contains("Start Stream")');
            const stopBtn = document.querySelector('button:contains("Stop Stream")');
            if (startBtn && stopBtn) {
                startBtn.addEventListener('click', startAutoRefresh);
                stopBtn.addEventListener('click', stopAutoRefresh);
                observer.disconnect();
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    if (document.readyState === 'complete') {
        setupAutoRefresh();
    } else {
        window.addEventListener('load', setupAutoRefresh);
    }
    </script>
    """)

if __name__ == "__main__":
    # Initialize the model before launching the interface
    init_model()
    
    iface.queue()  # Enable queuing for better performance
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=3000,
        favicon_path=None,
        show_error=True,
        quiet=True
    ) 