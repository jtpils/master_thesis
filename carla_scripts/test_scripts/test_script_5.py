#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import pandas as pd

import random
import time
import csv


def main():
    actor_list = []

    # In this test script we want to investigate how waypoints work.

    try:

        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle'))
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
        transform = random.choice(world.get_map().get_spawn_points())

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        vehicle.set_autopilot(False)  # SHUT DOWN AUTOPILOT

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.4)) # investigate which transform that is suitable
        myLidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(myLidar)
        print('created %s' % myLidar.type_id)
        myLidar.range = 30
        print('LiDAR range set to: ', myLidar.range)
        print(' ')

        '''i = 0
        while i<10:
            print('Vehicle location: ', vehicle.get_transform().location.x, vehicle.get_transform().location.y, vehicle.get_transform().location.z)
            transform = random.choice(world.get_map().get_spawn_points())
            print('random transform: ', transform.location.x, transform.location.y, transform.location.z)
            print('teleporting the car...')
            vehicle.set_transform(transform)
            time.sleep(5)
            print('Vehicle location: ', vehicle.get_transform().location.x, vehicle.get_transform().location.y, vehicle.get_transform().location.z)
            print(' ')
            i = i + 1
            time.sleep(3)'''



        # Create a list with the waypoints that exists
        map = world.get_map()
        waypoint_list = map.generate_waypoints(300.0)
        print('waypoint_list length:', len(waypoint_list))



        '''
        waypoint_list2 = map.generate_waypoints(3.0)
        print('waypoint_list length:', len(waypoint_list2))

        
        for element in waypoint_list[0:50]:

            print(element.road_id, element.transform.location.x, element.transform.location.y, element.transform.location.z)


        print(' ')

        #
        for element in waypoint_list2[0:50]:

            print(element.road_id, element.transform.location.x, element.transform.location.y, element.transform.location.z)


        # Do some kind of for loop that loops that puts the vehicle at the way point and lets it drive for some time.
        '''

        for waypoint_location in waypoint_list:
            
            print('waypoint location:' , waypoint_location.transform.location.x, waypoint_location.transform.location.y, waypoint_location.transform.location.z)
            vehicle.set_transform(waypoint_location.transform)
            time.sleep(0.1)

            print('vehicle current location:', vehicle.get_transform().location.x, vehicle.get_transform().location.y, vehicle.get_transform().location.z)
            print(' ')
            time.sleep(1)









    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')


if __name__ == '__main__':

    main()
