close all; clear; clc;

robot = importrobot('kinova_without_gripper.urdf');
robot.DataFormat = 'col';

load('curobo_trajectory.mat');
q = double(q);

% world_file = '../src/curobo/content/configs/world/simple_scenario.yml';

figure;
for i = 1:10:size(q,1)
    clf;
    show(robot, q(i,:)');
    axis off;
    box on;

    pause(dt);
end

% function h = plot_box(center, size)
% 
% end
