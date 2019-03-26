% THIS VERSION WORKS!

clear all;clc
% CHANGE PATH HERE
pc_path = '/Users/sabinalinderoth/Desktop/Town01_190308/pc/';
csv_path = '/Users/sabinalinderoth/Desktop/Town01_190308/Town01_190308.csv';
global_coordinates = importdata(csv_path);

files = dir(strcat(pc_path,'/*.ply'));

delimiterIn = ' ';
headerlinesIn = 7;

figure('units','normalized','outerposition',[0 0 1 1]);
%row = 1;
step = 1; % set to 1 if you want to visualize all frames
for file_number=1:step:length(files)-3
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
        temp.data(:,3) = -temp.data(:,3); % z

        % rotate and transform
        % find correct row:
        frame_number = str2double(file_name(1:end-4));
        row = find(global_coordinates.data(:,1)==frame_number);
        yaw = global_coordinates.data(row,5); % yaw in degrees
        if(yaw < 0)
            yaw = yaw + 360;
        end
       
        yaw = yaw + 90; 
        
        % lets rotate the point cloud
        Rz = [cosd(yaw) -sind(yaw) 0; sind(yaw) cosd(yaw) 0; 0 0 1];
        pc = Rz*temp.data';
        
        pc(2,:) = -pc(2,:); % y = -y
      

        global_coordinates.data(row,3) = - global_coordinates.data(row,3); % y = -y
        pc = pc + global_coordinates.data(row,2:4)';

        title('3D plot')
        hold on
        plot3(global_coordinates.data(row,2), global_coordinates.data(row,3), global_coordinates.data(row,4),'rd','MarkerSize',10)
        plot3(pc(1,:), pc(2,:), pc(3,:),'b.')
        xlabel('x')
        ylabel('y')
        zlabel('z')
        axis('equal')
  
    end
    
    pause(0.01)
end