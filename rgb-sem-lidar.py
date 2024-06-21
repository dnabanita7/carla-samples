import glob
import os
import sys
import random
import time
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 640
IM_HEIGHT = 480

rgb_data = []
sem_data = []
lidar_data = []
rgb_count = 0
sem_count = 0

def process_img(image):
    i = np.array(image.raw_data)
    global rgb_count
    rgb_count = rgb_count + 1

    #print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    rgb_data.append(i3)
    cv2.imwrite(f'sample_imgs/rgb_{rgb_count:03d}.png',i3)
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return rgb_data


def sem_callback(image):
    sem_image = np.frombuffer(image.raw_data, dtype=np.uint8)
    sem_image = np.reshape(sem_image, (image.height, image.width, 4))
    global sem_count 
    sem_count += 1
    sem_image = sem_image[:, :, 2]  # Only keep semantic segmentation labels
    sem_data.append(sem_image)
    cv2.imwrite(f'sample_imgs/sem_{sem_count:03d}.png', sem_image)
    cv2.imshow("", sem_image)
    cv2.waitKey(1)
    return sem_data

def lidar_callback(point_cloud):
    lidar_data.append(np.copy(point_cloud.raw_data))
    return lidar_data

actor_list = []


try:
    for _ in range(100):
        
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)

        world = client.get_world()
        


        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)


        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)


        world.tick()
        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.filter("model3")[0]
        print(bp)

        spawn_point = random.choice(world.get_map().get_spawn_points())

        vehicle = world.spawn_actor(bp, spawn_point)
        # vehicle.set_autopilot(True)

        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        actor_list.append(vehicle)

        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", f"110")

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
        actor_list.append(sensor)
        sensor.listen(lambda data: process_img(data))

        

        # Attach Semantic Segmentation camera to the vehicle
        sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', '800')
        sem_bp.set_attribute('image_size_y', '600')
        sem_bp.set_attribute('fov', '90')
        sem_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        sem_camera = world.spawn_actor(sem_bp, sem_transform, attach_to=vehicle)
        actor_list.append(sem_camera)
        sem_camera.listen(lambda data: sem_callback(data))
        

        # Attach LiDAR to the vehicle
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(lidar_sensor)
        lidar_sensor.listen(lambda data: lidar_callback(data))


        time.sleep(0.05)

finally:

    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")




