% visualize pointpillars

clear all;clc
%% ONE SWEEP

pc_path = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/pc/';
csv_path = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/test_set.csv';

global_coordinates = importdata(csv_path);

files = dir(strcat(pc_path,'/*.ply'));

delimiterIn = ' ';
headerlinesIn = 7;

% figure('units','normalized','outerposition',[0 0 1 1]);
step = 1; % set to 1 if you want to visualize all frames
file_number = 1;
file_name = files(file_number).name;
big_pc = [];

for file_number=205:205%215 %209:209
    file_name = files(file_number).name;
    
    path_to_file = strcat(pc_path, file_name);
    temp = importdata(path_to_file, delimiterIn, headerlinesIn);
    
    if isempty(temp.data) == 0  % if the file contains any points
        % keep all values that are smaller than 2, above ground
        %keep = temp.data(:,3) < 1.5;
        %temp.data = temp.data(keep,:);
        % keep all values that are larger than -10, to remove some detections upwards
        keep = temp.data(:,3 )> -5;
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
        
        % trim pc
        points_in_range = abs(pc(1,:))<15; %x
        pc = pc(:,points_in_range);
        points_in_range = abs(pc(2,:))<15; %y
        pc = pc(:,points_in_range);
        
        global_coordinates.data(row,3) = - global_coordinates.data(row,3); % y = -y
        %pc = pc + global_coordinates.data(row,2:4)';
        
        % plot3(global_coordinates.data(row,2), global_coordinates.data(row,3), global_coordinates.data(row,4),'rd','MarkerSize',10)
        % plot3(pc(1,:), pc(2,:), pc(3,:),)%'b.')
        
        big_pc = [big_pc; pc'];
        
    end
    
    
    
end



ax = pcshow(big_pc, 'MarkerSize', 80);
grid_size = 8;
axes(ax)
hold on
for i=-15:grid_size:15
    plot3(-15:grid_size:15, i*ones(length(-15:grid_size:15)), zeros(size(-15:grid_size:15)), 'r','LineWidth',3)
    plot3(i*ones(length(-15:grid_size:15)), -15:grid_size:15, zeros(size(-15:grid_size:15)), 'r','LineWidth',3)
    %plot3(xgrids(i,:), ygrids(i,:), zgrids(i,:),'r')
    %plot3(xgrids(:,i), ygrids(:,i), zgrids(i,:),'r')
end
plot3([1,1],[1,1],[0,5],'r','LineWidth',3)
plot3([1,1],[9,9],[0,5],'r','LineWidth',3)
plot3([9,9],[9,9],[0,5],'r','LineWidth',3)
plot3([9,9],[1,1],[0,5],'r','LineWidth',3)
plot3([1,1],[1,9],[5,5],'r','LineWidth',3)
plot3([1,9],[9,9],[5,5],'r','LineWidth',3)
plot3([9,9],[9,1],[5,5],'r','LineWidth',3)
plot3([9,1],[1,1],[5,5],'r','LineWidth',3)

