import glob
import os
import sys
import time
import random
import weakref

try:
    sys.path.append(
        glob.glob(
            os.path.join(
                os.path.dirname(__file__),
                "..", "carla", "dist",
                f"carla-*{sys.version_info.major}.{sys.version_info.minor}-win-amd64.egg"
            )
        )[0]
    )
except IndexError:
    raise RuntimeError(
        "Could not find CARLA .egg for your Python version.\n"
        "Make sure you are running this from WindowsNoEditor\\PythonAPI\\examples\n"
        "and that Python version matches the egg in PythonAPI\\carla\\dist"
    )

import carla

USE_PYGAME = True
try:
    import pygame
    import numpy as np
except Exception:
    USE_PYGAME = False


def make_camera_blueprint(world, width=1280, height=720, fov=90.0):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
  
    return cam_bp


def camera_callback(image, data_dict):
    """Store last image. Also optionally show/save."""
    data_dict["frame"] = image.frame
    data_dict["timestamp"] = image.timestamp

 
    if USE_PYGAME:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        data_dict["rgb"] = rgb
    else:
    
        if image.frame % 50 == 0:
            out_dir = os.path.join(os.path.dirname(__file__), "_out")
            os.makedirs(out_dir, exist_ok=True)
            image.save_to_disk(os.path.join(out_dir, "rgb_%06d.png" % image.frame))


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

   
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    actor_list = []
    try:
      
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in this map.")
        spawn_point = random.choice(spawn_points)

      
        bp_lib = world.get_blueprint_library()
        vehicle_bp = random.choice(bp_lib.filter("vehicle.*model3*")) if bp_lib.filter("vehicle.*model3*") else random.choice(bp_lib.filter("vehicle.*"))
        vehicle_bp.set_attribute("role_name", "hero")

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
          
            for sp in spawn_points[:20]:
                vehicle = world.try_spawn_actor(vehicle_bp, sp)
                if vehicle:
                    break
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle. Try restarting CARLA or choose another map.")

        actor_list.append(vehicle)
        print(f"[OK] Spawned vehicle: {vehicle.type_id} (id={vehicle.id})")

       
        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(vehicle, 0) 
        traffic_manager.distance_to_leading_vehicle(vehicle, 2.5)

       
        cam_bp = make_camera_blueprint(world, width=1280, height=720, fov=90)
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  

        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(camera)
        print(f"[OK] Spawned camera sensor (id={camera.id})")

       
        cam_data = {"rgb": None, "frame": None, "timestamp": None}
        weak_cam_data = weakref.ref(cam_data)
        camera.listen(lambda img: camera_callback(img, weak_cam_data()))

        if USE_PYGAME:
            pygame.init()
            w, h = 1280, 720
            screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("CARLA - my_vehicle_demo (ESC to quit)")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("Arial", 18)
            print("[INFO] Pygame viewer enabled.")
        else:
            print("[INFO] Pygame/numpy not available. Will save an image every ~50 frames into examples/_out/")

        print("[INFO] Running... Press ESC (pygame) or Ctrl+C to quit.")

     
        start_time = time.time()
        while True:
            world.tick()

          
            if USE_PYGAME:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return

                rgb = cam_data.get("rgb", None)
                if rgb is not None:
               
                    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                    screen.blit(surface, (0, 0))

                    
                    v = vehicle.get_velocity()
                    speed = (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5 * 3.6  
                    frame = cam_data.get("frame")
                    t = time.time() - start_time
                    txt = f"Frame: {frame} | Speed: {speed:5.1f} km/h | t={t:5.1f}s"
                    overlay = font.render(txt, True, (255, 255, 255))
                    screen.blit(overlay, (10, 10))

                pygame.display.flip()
                clock.tick(60)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
      
        print("[INFO] Cleaning up actors...")
        try:
            # Stop sensors first
            for a in actor_list:
                if "sensor" in a.type_id:
                    a.stop()
        except Exception:
            pass

        for a in actor_list:
            try:
                a.destroy()
            except Exception:
                pass


        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        if USE_PYGAME:
            try:
                pygame.quit()
            except Exception:
                pass

        print("[OK] Done.")


if __name__ == "__main__":
    main()
