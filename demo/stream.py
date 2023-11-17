import os
import time
import queue
import traceback
import threading

# PyPI Packages
import cv2 as cv
import olympe
import json
import tritonclient.http as httpclient
from tritonclient.utils import *
import visualization_utils as viz_utils
from PIL import Image
import numpy as np

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})
DRONE_IP = os.environ.get("DRONE_IP", "192.168.53.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")
 # CONSTANT VARIBLES TAKEN FROM rundetector.py
DETECTOR_METADATA = {
    'v5a.0.0': {
        'megadetector_version':'v5a.0.0',
        'typical_detection_threshold':0.2,
        'conservative_detection_threshold':0.05
    },
    'v5b.0.0': {
        'megadetector_version':'v5b.0.0',
        'typical_detection_threshold':0.2,
        'conservative_detection_threshold':0.05
    }
}
DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = DETECTOR_METADATA['v5b.0.0']['typical_detection_threshold']
DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.005
DEFAULT_BOX_THICKNESS = 4
DEFAULT_BOX_EXPANSION = 0
DEFAULT_DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'
}

class VideoStream(threading.Thread):
    def __init__(self) -> None:
        self.drone = olympe.Drone(DRONE_IP)
        self.frame_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()
        self.client = httpclient.InferenceServerClient(url="172.17.0.2:8000")
        self.frame_counter = 0
        super().__init__()
        super().start()
        
    def start(self):
        self.drone.connect()
        self.drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
            # h264_cb=self.h264_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )
        # Start video streaming
        self.drone.streaming.start()
    
    def stop(self):
        self.drone.streaming.stop()
        self.drone.disconnect()

    def yuv_frame_cb(self, yuv_frame):
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)
    
    def flush_cb(self, _):
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True
    def start_cb(self):
        pass
    
    def end_cb(self):
        pass

    def show_yuv_frame(self, window_name, cv_frame):
        cv.imshow(window_name, cv_frame)
        cv.waitKey(1)
    
    def send_to_triton(self, cv_frame):
        input_tensor = [httpclient.InferInput("image", cv_frame.shape, datatype="UINT8")]
        input_tensor[0].set_data_from_numpy(cv_frame)

        output = [httpclient.InferRequestedOutput("detection_result", binary_data=False)]

        query_response = self.client.infer(model_name="MegaDetector", model_version="1", inputs=input_tensor, outputs=output)

        triton_output = query_response.as_numpy("detection_result")

        result = json.loads( triton_output[0])

        image = Image.fromarray(cv_frame)

        viz_utils.render_detection_bounding_boxes(
            result['detection'], 
            image,
            label_map=DEFAULT_DETECTOR_LABEL_MAP,
            confidence_threshold=DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
            thickness=DEFAULT_BOX_THICKNESS,
            expansion=DEFAULT_BOX_EXPANSION
        )

        return np.array(image)
    
    def to_cv_frame(self, yuv_frame):
        cvt_color_flag = {
            olympe.VDEF_I420: cv.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv.COLOR_YUV2BGR_NV12,
        }[yuv_frame.format()]

        cv_frame = cv.cvtColor(yuv_frame.as_ndarray(), cvt_color_flag)
        return cv_frame

    def run(self):
        window_name = "HackOhio"
        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
        main_thread = next(
            filter(lambda t: t.name == "MainThread", threading.enumerate())
        )
        while self.flush_queue_lock:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.01)
                self.frame_counter += 1
            except queue.Empty:
                continue
            try:
                cv_frame = self.to_cv_frame(yuv_frame)
                if self.frame_counter % 3 == 0:
                    cv_frame = self.send_to_triton(cv_frame)
                self.show_yuv_frame(window_name, cv_frame)
            except Exception:
                traceback.print_exc()
            finally:
                yuv_frame.unref()
        cv.destroyWindow(window_name)
if __name__ == "__main__":
    cv.startWindowThread()
    
    stream = VideoStream()
    stream.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    stream.stop()
