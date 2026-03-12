import carla
import socket
import struct
import cv2
import numpy as np
import time

JETSON_IP = "192.168.0.102"
JETSON_PORT = 5001

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

IMG_W = 640
IMG_H = 360
JPEG_QUALITY = 80

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((JETSON_IP, JETSON_PORT))
print(f"[PC] Connected to Jetson {JETSON_IP}:{JETSON_PORT}")

client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

vehicle_bp = bp_lib.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

if vehicle is None:
    raise RuntimeError("Could not spawn vehicle")

camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(IMG_W))
camera_bp.set_attribute("image_size_y", str(IMG_H))
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", "0.1")

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

actors = [camera, vehicle]

def send_frame(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame_bgr = array[:, :, :3]

    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return

    data = encoded.tobytes()
    header = struct.pack("!IdI", image.frame, image.timestamp, len(data))
    sock.sendall(header)
    sock.sendall(data)

    print(f"[PC] sent frame={image.frame}")

camera.listen(send_frame)

try:
    print("[PC] Streaming... Ctrl+C to stop")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    for actor in actors:
        if actor is not None:
            actor.destroy()
    sock.close()
    print("[PC] Clean exit")
