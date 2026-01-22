import pyrealsense2 as rs
import numpy as np
import time
import cv2

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.running = False

    def check_connected_cameras(self):
        ctx = rs.context()
        connected_cameras = []
        for device in ctx.query_devices():
            serial_number = device.get_info(rs.camera_info.serial_number)
            connected_cameras.append(serial_number)
        return connected_cameras

    def start_pipeline(self, cameras=None):
        if cameras is None or len(cameras) == 0:
            print("No cameras specified.")
            return

        for cam_serial in cameras:
            self.config.enable_device(cam_serial)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.running = True
        # Discard first few frames to allow auto-exposure to settle
        for _ in range(5):
            self.pipeline.wait_for_frames()

    def stop_pipeline(self):
        if self.running:
            self.pipeline.stop()
            self.running = False

    def save_rgb_frame(self, cameras=None, path=None):
        if not self.running:
            print("Pipeline is not running!")
            return

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("No color frame available.")
                return

            color_image = np.asanyarray(color_frame.get_data())

            if path is None:
                path = f"rgb_frame_{int(time.time())}.png"

            cv2.imwrite(path, color_image)
            print(f"Saved RGB frame to {path}")

        except Exception as e:
            print(f"Error capturing frame: {e}")

    def get_intrinsic_and_distortion(self, camera_serial):
        ctx = rs.context()
        for device in ctx.query_devices():
            serial_number = device.get_info(rs.camera_info.serial_number)
            if serial_number == camera_serial:
                sensor = device.first_color_sensor()
                intrinsics = sensor.get_stream_profiles()[0].as_video_stream_profile().get_intrinsics()
                intrinsic_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                             [0, intrinsics.fy, intrinsics.ppy],
                                             [0, 0, 1]])
                distortion_coeffs = np.array(intrinsics.coeffs)
                return intrinsic_matrix, distortion_coeffs
        print(f"Camera with serial {camera_serial} not found.")
        return None, None

if __name__ == "__main__":
    camera = RealSenseCamera()
    print("Connected cameras:", camera.check_connected_cameras())