import carla
import socket
import struct
import cv2
import numpy as np
import time

JETSON_IP = "192.168.0.102"
JETSON_PORT = 5001

IMG_W = 320
IMG_H = 240
JPEG_QUALITY = 60

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

print("[PC] Connecting to existing CARLA world")
world = client.get_world()
print("[PC] Current map:", world.get_map().name)

blueprints = world.get_blueprint_library()

vehicle_bp = blueprints.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()

vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

if vehicle is None:
    raise RuntimeError("Vehicle spawn failed")

vehicle.set_autopilot(True)
print("[PC] Vehicle spawned")
print("[PC] Autopilot enabled")

camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(IMG_W))
camera_bp.set_attribute("image_size_y", str(IMG_H))
camera_bp.set_attribute("sensor_tick", "0.5")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

actors = [camera, vehicle]

def send_frame(image):
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame_bgr = array[:, :, :3]

        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

        if not ok:
            return

        data = encoded.tobytes()
        header = struct.pack("!IdI", image.frame, image.timestamp, len(data))
        packet = header + data

        sock.sendto(packet, (JETSON_IP, JETSON_PORT))

        print(f"[PC] sent frame {image.frame}")

    except Exception as e:
        print("[PC] send error:", e)

camera.listen(send_frame)

try:
    print("[PC] Streaming to Jetson... Ctrl+C to stop")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n[PC] Stopping...")

finally:
    try:
        camera.stop()
    except Exception:
        pass

    for actor in actors:
        try:
            actor.destroy()
        except Exception:
            pass
