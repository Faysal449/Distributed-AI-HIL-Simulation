import carla
import socket
import struct
import json
import cv2
import numpy as np
import time
import threading
import queue
import csv

JETSON_IP = "192.168.0.102"
FRAME_PORT = 5001
FEEDBACK_PORT = 5002

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

IMG_W = 320
IMG_H = 240
JPEG_QUALITY = 60
TEST_FRAMES = 100

frame_queue = queue.Queue(maxsize=1)
running = True
send_times = {}
rows = []
sent_frames = 0

def feedback_server():
    global running
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", FEEDBACK_PORT))
    server.listen(1)
    print(f"[PC-TCP] Waiting for feedback on port {FEEDBACK_PORT}...")

    conn, addr = server.accept()
    print(f"[PC-TCP] Feedback connected from {addr}")

    buffer = b""
    try:
        while running:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                data = json.loads(line.decode("utf-8"))

                pc_feedback_time = time.time()
                frame_id = data["frame"]
                pc_send_time = data["pc_send_time"]

                rtt_ms = (pc_feedback_time - pc_send_time) * 1000.0

                row = {
                    "protocol": "tcp",
                    "frame": frame_id,
                    "pc_send_time": pc_send_time,
                    "jetson_recv_time": data["jetson_recv_time"],
                    "jetson_done_time": data["jetson_done_time"],
                    "pc_feedback_time": pc_feedback_time,
                    "inference_ms": data["inference_ms"],
                    "round_trip_ms": round(rtt_ms, 3),
                    "count": data["count"],
                }
                rows.append(row)
                print(f"[TCP] frame={frame_id} RTT={rtt_ms:.2f} ms inf={data['inference_ms']:.2f} ms")
    finally:
        conn.close()
        server.close()

def sensor_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame_bgr = array[:, :, :3].copy()

    item = (int(image.frame), frame_bgr)

    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass

    frame_queue.put_nowait(item)

def save_csv(filename, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def main():
    global running, sent_frames

    feedback_thread = threading.Thread(target=feedback_server, daemon=True)
    feedback_thread.start()
    time.sleep(1.0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((JETSON_IP, FRAME_PORT))
    print(f"[PC-TCP] Connected to Jetson {JETSON_IP}:{FRAME_PORT}")

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    print("[PC-TCP] Current map:", world.get_map().name)

    blueprints = world.get_blueprint_library()
    vehicle_bp = blueprints.filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

    if vehicle is None:
        raise RuntimeError("Vehicle spawn failed")

    vehicle.set_autopilot(True)
    print("[PC-TCP] Vehicle spawned, autopilot enabled")

    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMG_W))
    camera_bp.set_attribute("image_size_y", str(IMG_H))
    camera_bp.set_attribute("sensor_tick", "0.5")
    camera_bp.set_attribute("fov", "90")

    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle
    )

    actors = [camera, vehicle]
    camera.listen(sensor_callback)

    try:
        while sent_frames < TEST_FRAMES:
            try:
                frame_id, frame_bgr = frame_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            ok, encoded = cv2.imencode(
                ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                continue

            data = encoded.tobytes()
            pc_send_time = time.time()
            header = struct.pack("!IdI", frame_id, pc_send_time, len(data))

            sock.sendall(header)
            sock.sendall(data)

            send_times[frame_id] = pc_send_time
            sent_frames += 1
            print(f"[PC-TCP] sent frame={frame_id}")

        print("[PC-TCP] Finished sending test frames, waiting 5 seconds for feedback...")
        time.sleep(5)

    finally:
        running = False
        try:
            camera.stop()
        except Exception:
            pass
        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            sock.close()
        except Exception:
            pass

        save_csv("tcp_latency_results.csv", rows)
        print("[PC-TCP] Saved tcp_latency_results.csv")

if __name__ == "__main__":
    main()
