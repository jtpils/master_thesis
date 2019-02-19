##############################################################################################################################
# This is a modified version of our gathering_data.py-script. After feedback from Lennart, we will gather dta without using the autopilot. A better (or at least as good?) approach would be to manually move the car and save a lidar sweep at each new position. 

# Find positions using waypoints, approximately at 1 meters distance.
# in the callback function for the lidar, include som if-statement, such that we only save the measurement if some conditions in fulfilled. This condition can be that we have moved 1 m since the last measurement (the car is at its new position) or that some time has passed (maybe every tenth frame or so).
# save some rgb-images even less frequently.
##############################################################################################################################


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
import random
import time
import csv


# Define global variable for storing the latest position of the LiDAR
lidar_position = np.array((0,0))

def main():

    input_folder_name = input('Type name of new folder: "TownXX_date_number" :')
    #print(input_folder_name)
    #print(type(input_folder_name))

    folder_name = '/_out_' + input_folder_name

    # creates folder to store the lidar data
    current_path = os.getcwd()
    folder_path = current_path + folder_name

    try:
       os.mkdir(folder_path)
    except OSError:
       print('Failed to create new directory.')
    else:
       print('Successfully created new directory with path: ' , folder_path)



    actor_list = []

    try:

        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.toyota.prius')
        transform = random.choice(world.get_map().get_spawn_points()) # do we want to set it at the same location every time?
        #print('random transform: ', transform.location.x, )
        vehicle = world.spawn_actor(vehicle_bp, transform)
        actor_list.append(vehicle)
        print(' ')
        print('created %s' % vehicle.type_id)
        #vehicle.set_autopilot(True)
        #print('Auto pilot')
        time.sleep(3) # let the car have some fun
        print(' ')


        ########################################## add sensors ##########################################
        # Add camera
        # Find the blueprint of the sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # Modify attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)


        # Add LiDAR
        # transform we give here is now relative to the vehicle.
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        #lidar_bp.set_attribute('points_per_second', '5000000')
        #lidar_bp.set_attribute('channels', '64')
        #lidar_bp.set_attribute('upper_fov', '20')
        #lidar_bp.set_attribute('lower_fov', '-30')
        lidar_bp.set_attribute('range', '10000')  # this one is important.

        lidar_transform = carla.Transform(carla.Location(z=2.2)) # investigate which transform that is suitable
        myLidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(myLidar)
        print('created %s' % myLidar.type_id)
        print('LiDAR attributes: ', myLidar.attributes)
        print(' ')

        ########################################## create csv-file ##########################################


        # Creates the csv file
        csv_path = folder_path + '/' + input_folder_name + '.csv'
        with open(csv_path , mode = 'w') as csv_file:
            fieldnames = ['frame_number', 'x', 'y', 'z', 'yaw']
            csv_writer = csv.writer(csv_file, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(fieldnames)


        ########################################## callback functions ##########################################
        def save_lidar_data(data):
            global lidar_position
            current_position = np.array((data.transform.location.x, data.transform.location.y))
            moved_distance = np.linalg.norm(current_position - lidar_position)
        
            if moved_distance > 0.5 and data.frame_number % 10 == 0:  # moved at least 0.5 m, and has passed 1 second (so that we don't save sweeps when the car is dropped)
                point_cloud_path = folder_path + '/pc/%06d'
                data.save_to_disk( point_cloud_path % data.frame_number)

                # open and append data to the csv file
                with open(csv_path , mode = 'a') as csv_file_2:
                    csv_writer_2 = csv.writer(csv_file_2, delimiter = ',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer_2.writerow([data.frame_number, data.transform.location.x, data.transform.location.y, data.transform.location.z, data.transform.rotation.yaw])
            
                lidar_position = current_position  # save globally for next lidar sweep
            

        '''def save_rgb_data(image):
            if image.frame_number % 100 == 0:  # every 100th frame, i.e. once every 10 second
                rgb_path = folder_path + '/rgb/%06d'
                data.save_to_disk( rgb_path % image.frame_number)'''
                
                
        #camera.listen(lambda image: save_rgb_data(image))
        myLidar.listen(lambda data: save_lidar_data(data)) # data is a LidarMeasurement object
        
        ########################################## main loops ##########################################
        

        # Create a list with the waypoints that exists
        map = world.get_map()
        waypoint_list = map.generate_waypoints(1) 
        print('waypoint list length:', len(waypoint_list))
        print(' ')

        num_waypoints = 1
        for waypoint in waypoint_list:  ###### CHANGED HERE
            vehicle.set_transform(waypoint.transform)
            time.sleep(2) # seconds to move 

            if num_waypoints%5 == 0:
                text = 'number of waypoints visited: ' + str(num_waypoints) + ' of ' + str(len(waypoint_list))
                print(text)
            num_waypoints = num_waypoints + 1

        
        

    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')



if __name__ == '__main__':

    main()
    
    
# Where should we add code related to processing of the data after the carla-part is over? Here? or should we close the carla client and then start another script?

