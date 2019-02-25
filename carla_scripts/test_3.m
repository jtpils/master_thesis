% plots the LiDAR position and point cloud, but removes detections on the
% ground.

clear all;clc
% CHANGE PATH HERE
%pc_path = './_out_position/pc/';
%csv_path = './_out_position/position.csv';
pc_path = '/home/master04/Desktop/_out_Town02_190221_1/pc/';
csv_path = '/home/master04/Desktop/_out_Town02_190221_1/Town02_190221_1.csv';
global_coordinates = importdata(csv_path);

%files = dir(strcat(pc_path,'/*.ply'));
files = dir(strcat(pc_path,'/003121.ply'))
delimiterIn = ' ';
headerlinesIn = 7;

figure('units','normalized','outerposition',[0 0 1 1]);
row = 1;
step = 1; % set to 1 if you want to visualize all frames

for file_number=1:step:length(files)
    file_name = files(file_number).name;
    try
        path_to_file = strcat(pc_path, file_name);
        temp = importdata(path_to_file, delimiterIn, headerlinesIn);
        %disp(file_name)
    catch
        warning('Could not load file:')
        disp(file_name)
        continue
    end
    
    if isempty(temp.data) == 0  % if the file contains any points
        % keep all values that are smaller than 2, above ground
        keep = temp.data(:,3) < 1.5;
        temp.data = temp.data(keep,:);
        % keep all values that are larger than -10, to remove some detections upwards
        keep = temp.data(:,3 )> -11;
        temp.data = temp.data(keep,:);

        % transform to our preferred coordinate system
        temp.data(:,2) = -temp.data(:,2); % y
        temp.data(:,3) = -temp.data(:,3); % z

        % rotate and transform
        yaw = global_coordinates.data(row,5) + 90; % yaw in degrees
        
        % lets rotate the point cloud
        Rz = [cosd(yaw) -sind(yaw) 0; sind(yaw) cosd(yaw) 0; 0 0 1];
        pc = Rz*temp.data';
        
        % now translate to global coordinates
        pc = pc + global_coordinates.data(row,2:4)';
        
        % plot
       
%         subplot(1,2,1);
%         title('3D plot')
%         hold on
%         plot3(global_coordinates.data(1:row,2), global_coordinates.data(1:row,3), global_coordinates.data(1:row,4),'k*','MarkerSize',10)
%         plot3(pc(1,:), pc(2,:), pc(3,:),'b.')
%         xlabel('x')
%         ylabel('y')
%         zlabel('z')
%         axis('equal')
%         %hold off
        
        %subplot(1,2,2)
        hold on
        plot(pc(1,:), pc(2,:),'b.')
        plot(global_coordinates.data(1:row,2), global_coordinates.data(1:row,3),'k.','MarkerSize',10)
        title('2D plot')
        xlabel('x')
        ylabel('y')
        grid on
        axis('equal')
        %hold off
        
    end
    row = row + step;
    
    pause(0.01)
end
