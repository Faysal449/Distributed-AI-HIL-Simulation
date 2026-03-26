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
TM_PORT = 8000

IMG_W = 320
IMG_H = 240
JPEG_QUALITY = 60
TEST_FRAMES = 100
CAMERA_TICK = 0.1

HEADER_FORMAT = "!IdI"

frame_queue = queue.Queue(maxsize=1)
running = True
rows = []
sent_frames = 0
pc_dropped_frames = 0


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

                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception as e:
                    print(f"[PC-TCP] Bad feedback packet: {e}")
                    continue

                pc_feedback_time = time.time()
                frame_id = data["frame"]
                rtt_ms = (pc_feedback_time - data["pc_send_time"]) * 1000.0

                row = {
                    "protocol": "tcp",
                    "frame": frame_id,
                    "pc_send_time": data["pc_send_time"],
                    "jetson_recv_time": data["jetson_recv_time"],
                    "jetson_done_time": data["jetson_done_time"],
                    "pc_feedback_time": pc_feedback_time,
                    "inference_ms": round(data["inference_ms"], 3),
                    "round_trip_ms": round(rtt_ms, 3),
                    "count": data["count"],
                    "jpeg_bytes": data.get("jpeg_bytes", 0),
                }
                rows.append(row)

                print(
                    f"[TCP] frame={frame_id} "
                    f"RTT={rtt_ms:.2f} ms "
                    f"inf={data['inference_ms']:.2f} ms "
                    f"count={data['count']}"
                )

    except Exception as e:
        print(f"[PC-TCP] Feedback server error: {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        server.close()


def sensor_callback(image):
    global pc_dropped_frames

    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame_bgr = array[:, :, :3].copy()

        item = (int(image.frame), frame_bgr)

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
                pc_dropped_frames += 1
            except queue.Empty:
                pass

        frame_queue.put_nowait(item)

    except Exception as e:
        print(f"[PC-TCP] sensor_callback error: {e}")


def save_csv(filename, rows):
    keys = [
        "protocol",
        "frame",
        "pc_send_time",
        "jetson_recv_time",
        "jetson_done_time",
        "pc_feedback_time",
        "inference_ms",
        "round_trip_ms",
        "count",
        "jpeg_bytes",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        else:
            print("[PC-TCP] WARNING: No feedback rows, saved empty CSV with header only")


def main():
    global running, sent_frames

    vehicle = None
    camera = None
    actors = []

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

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(TM_PORT)
    traffic_manager.set_synchronous_mode(False)

    blueprints = world.get_blueprint_library()

    vehicle_bp = blueprints.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found")

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if vehicle is None:
        for sp in spawn_points[1:]:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break

    if vehicle is None:
        raise RuntimeError("Vehicle spawn failed")

    vehicle.set_autopilot(True, traffic_manager.get_port())
    print("[PC-TCP] Vehicle spawned, autopilot enabled")

    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMG_W))
    camera_bp.set_attribute("image_size_y", str(IMG_H))
    camera_bp.set_attribute("sensor_tick", str(CAMERA_TICK))
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.4),
        carla.Rotation(pitch=0.0)
    )

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actors = [camera, vehicle]

    camera.listen(sensor_callback)
    print("[PC-TCP] Camera started")
    print("[PC-TCP] Warming up camera for 2 seconds...")
    time.sleep(2.0)

    start_test_time = time.time()
    empty_queue_tries = 0
    max_empty_queue_tries = 40

    try:
        while sent_frames < TEST_FRAMES:
            try:
                frame_id, frame_bgr = frame_queue.get(timeout=1.0)
                empty_queue_tries = 0
            except queue.Empty:
                empty_queue_tries += 1
                print(f"[PC-TCP] No frame received from queue ({empty_queue_tries}/{max_empty_queue_tries})")

                if empty_queue_tries >= max_empty_queue_tries:
                    print("[PC-TCP] Too many empty queue timeouts, ending test early")
                    break
                continue

            ok, encoded = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                print(f"[PC-TCP] JPEG encode failed for frame={frame_id}")
                continue

            data = encoded.tobytes()
            pc_send_time = time.time()

            header = struct.pack(HEADER_FORMAT, frame_id, pc_send_time, len(data))
            sock.sendall(header)
            sock.sendall(data)

            sent_frames += 1

            print(
                f"[PC-TCP] sent frame={frame_id} "
                f"size={len(data)} bytes "
                f"sent_count={sent_frames}/{TEST_FRAMES}"
            )

        print("[PC-TCP] Finished sending test frames")
        print("[PC-TCP] Waiting 5 seconds for late feedback...")
        time.sleep(5)

    finally:
        running = False

        total_test_time = time.time() - start_test_time
        approx_send_fps = sent_frames / total_test_time if total_test_time > 0 else 0.0

        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        for actor in actors:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

        try:
            sock.close()
        except Exception:
            pass

        save_csv("tcp_latency_results.csv", rows)

        print("\n[PC-TCP] ===== TEST SUMMARY =====")
        print(f"[PC-TCP] Sent frames           : {sent_frames}")
        print(f"[PC-TCP] Feedback rows         : {len(rows)}")
        print(f"[PC-TCP] PC dropped frames     : {pc_dropped_frames}")
        print(f"[PC-TCP] Test duration         : {total_test_time:.2f} s")
        print(f"[PC-TCP] Approx send FPS       : {approx_send_fps:.2f}")
        print("[PC-TCP] Saved tcp_latency_results.csv")


if __name__ == "__main__":
    main()
