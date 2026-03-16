import carla
import socket
import struct
import cv2
import numpy as np
import time
import threading


JETSON_IP = "192.168.0.102"
FRAME_PORT = 5001
FEEDBACK_PORT = 5002


CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

IMG_W = 320
IMG_H = 240
JPEG_QUALITY = 70


def feedback_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", FEEDBACK_PORT))
    server.listen(1)

    print(f"[PC] Waiting for Jetson feedback on port {FEEDBACK_PORT}...")

    conn, addr = server.accept()
    print(f"[PC] Feedback connected from {addr}")

    buffer = b""

    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break

            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    print("[DETECTION]", line.decode("utf-8"))
                except:
                    pass
    finally:
        conn.close()
        server.close()

feedback_thread = threading.Thread(target=feedback_server, daemon=True)
feedback_thread.start()

time.sleep(1)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((JETSON_IP, FRAME_PORT))
print(f"[PC] Connected to Jetson {JETSON_IP}:{FRAME_PORT}")


client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(10.0)

print("[PC] Loading Town03...")

client.load_world("Town03")
world = client.get_world()

blueprints = world.get_blueprint_library()


vehicle_bp = blueprints.filter("vehicle.*")[0]

spawn_points = world.get_map().get_spawn_points()

vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

if vehicle is None:
    raise RuntimeError("Vehicle spawn failed")

print("[PC] Vehicle spawned")


camera_bp = blueprints.find("sensor.camera.rgb")

camera_bp.set_attribute("image_size_x", str(IMG_W))
camera_bp.set_attribute("image_size_y", str(IMG_H))
camera_bp.set_attribute("fov", "90")


camera_bp.set_attribute("sensor_tick", "0.2")

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

actors = [vehicle, camera]

print("[PC] Camera started")


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

        header = struct.pack(
            "!IdI",
            image.frame,
            image.timestamp,
            len(data)
        )

        sock.sendall(header)
        sock.sendall(data)

        print(f"[PC] sent frame {image.frame}")

    except Exception as e:
        print("[PC] send error:", e)

camera.listen(send_frame)


try:
    print("[PC] Streaming to Jetson... Press Ctrl+C to stop")

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n[PC] Stopping...")

finally:
    for actor in actors:
        if actor:
            actor.destroy()

    sock.close()

    print("[PC] Clean exit")
