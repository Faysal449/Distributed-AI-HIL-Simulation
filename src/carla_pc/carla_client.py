import sys
import socket
import struct
import time
import numpy as np
import cv2


sys.path.append(r"E:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-py3.7-win-amd64.egg")
import carla


JETSON_IP = "192.168.0.102"  
JETSON_PORT = 5000


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((JETSON_IP, JETSON_PORT))
print("[OK] Connected to Jetson")


client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()
print("[OK] Connected to CARLA")

settings = world.get_settings()
settings.synchronous_mode = False
settings.no_rendering_mode = True          
settings.fixed_delta_seconds = None
world.apply_settings(settings)
print("[OK] no_rendering_mode enabled")



actors = world.get_actors()
for a in actors:
    try:
        tid = a.type_id
        if tid.startswith("sensor.camera") or tid.startswith("sensor.lidar") or tid.startswith("sensor.other") or tid.startswith("vehicle."):
            a.destroy()
    except:
        pass
print("[OK] Old actors cleaned")


vehicle_bp = bp_lib.filter("vehicle.*")[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])

if vehicle is None:
    raise RuntimeError("Could not spawn vehicle (spawn point busy). Try restarting CARLA or choose another spawn point.")

print("[OK] Vehicle spawned")


W, H = 80, 60 
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(W))
camera_bp.set_attribute("image_size_y", str(H))
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", "0.2")  

cam_tf = carla.Transform(carla.Location(x=1.5, z=2.2))
camera = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)
print("[OK] Camera spawned (80x60 @ 5 FPS)")

def on_image(image: carla.Image):
  
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((H, W, 4))
    bgr = arr[:, :, :3]

  
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    payload = gray.tobytes()
    header = struct.pack("I", len(payload))

    try:
        sock.sendall(header + payload)
    except:
   
        try:
            camera.stop()
        except:
            pass

camera.listen(on_image)
print("[OK] Streaming started (grayscale)")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("[STOP] Ctrl+C received")
finally:
    try:
        camera.stop()
    except:
        pass
    try:
        camera.destroy()
    except:
        pass
    try:
        vehicle.destroy()
    except:
        pass
    try:
        sock.close()
    except:
        pass
    print("[OK] Clean exit")
