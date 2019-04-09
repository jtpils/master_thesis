% visulaize pillar features
clf
num_points=2;
%x = [-5*rand(1,num_points), 5*rand(1,num_points)];
%y = [-5*rand(1,num_points), 5*rand(1,num_points)];
%z = 20*rand(1,num_points*2);


plot3(x,y,z,'b*', 'MarkerSize',10)
hold on
xgrid = [-1, 1, 1, -1, -1]; xgrid = [xgrid, xgrid, 1, 1, 1,1,-1,-1];
ygrid = [-1, -1, 1, 1, -1]; ygrid = [ygrid, ygrid, -1, -1,1,1,1,1];
zgrid = [zeros(1,5), 20*ones(1,5), 20,0, 0,20,20,0];
xgrid = xgrid*3; ygrid = ygrid*3;
plot3(xgrid, ygrid, zgrid,'b')
axis([-5,5,-5,5,0,20])

plot3([0, 0],[0,0],[0,20],'k') %center line
% plot offset:
for i=1:4
    plot3([x(i),0],[y(i),0],[z(i),z(i)],'r')
end
xlabel('x')
ylabel('y')
zlabel('z')