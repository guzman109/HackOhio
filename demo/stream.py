import os
import time
import queue
import traceback
import threading

# PyPI Packages
import cv2 as cv
import olympe

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})
DRONE_IP = os.environ.get("DRONE_IP", "192.168.53.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")

class VideoStream(threading.Thread):
    def __init__(self) -> None:
        self.drone = olympe.Drone(DRONE_IP)
        self.frame_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()
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
    
    def flush_cb(self):
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True
    def start_cb(self):
        pass
    
    def end_cb(self):
        pass

    def show_yuv_frame(self, window_name, yuv_frame):
        cvt_color_flag = {
            olympe.VDEF_I420: cv.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv.COLOR_YUV2BGR_NV12,
        }[yuv_frame.format()]

        cv_frame = cv.cvtColor(yuv_frame.as_ndarray(), cvt_color_flag)

        cv.imshow(window_name, cv_frame)
        cv.waitKey(1)
    
    def run(self):
        window_name = "HackOhio"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        main_thread = next(
            filter(lambda t: t.name == "MainThread", threading.enumerate())
        )
        while self.flush_queue_lock:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue
            try:
                self.show_yuv_frame(window_name, yuv_frame)
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
