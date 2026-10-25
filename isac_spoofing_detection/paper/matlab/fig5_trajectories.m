%% Figure 5: Four UAV Swarm Trajectory Types with ISAC Anchors
clear; close all;

%% --- Trajectory parameters ---
% ISAC ground anchors (fixed)
anchors = [25 35 5; 25 70 25; 60 35 35; 85 70 15];

%% --- Generate representative trajectories ---
t = linspace(0, 60, 600)';  % 60s, 0.1s step
N = 8;  % 8 UAVs

% (a) S-curve
x_sc = @(t,uav) 30 + 30*sin(2*pi*t/60 + 2*pi*uav/N);
y_sc = @(t,uav) 40 + 25*sin(pi*t/30 + 2*pi*uav/N + pi/2);
z_sc = @(t,uav) 20 + 10*sin(2*pi*t/30 + uav);

% (b) Straight
x_st = @(t,uav) 10 + uav*8 + 0.5*t;
y_st = @(t,uav) 20 + uav*5;
z_st = @(t,uav) 25 + uav*2;

% (c) Circle
x_ci = @(t,uav) 50 + 25*cos(2*pi*t/30 + 2*pi*uav/N);
y_ci = @(t,uav) 50 + 25*sin(2*pi*t/30 + 2*pi*uav/N);
z_ci = @(t,uav) 25 + 5*cos(2*pi*t/20 + uav);

% (d) Helix
x_he = @(t,uav) 40 + 20*cos(2*pi*t/40 + 2*pi*uav/N);
y_he = @(t,uav) 45 + 20*sin(2*pi*t/40 + 2*pi*uav/N);
z_he = @(t,uav) 15 + 10*t/60 + 3*sin(2*pi*t/15 + uav);

traj = struct('name',{'S-Curve','Straight','Circle','Helix'}, ...
              'x',{x_sc,x_st,x_ci,x_he}, ...
              'y',{y_sc,y_st,y_ci,z_he}, ...
              'z',{z_sc,z_st,z_ci,z_he});

figure('Position',[50 50 1200 900]);
for k = 1:4
    subplot(2,2,k);
    hold on;
    colors = lines(N);
    for uav = 0:N-1
        xx = traj(k).x(t, uav);
        yy = traj(k).y(t, uav);
        zz = traj(k).z(t, uav);
        plot3(xx, yy, zz, 'Color', colors(uav+1,:), 'LineWidth',1.2);
        % start marker
        plot3(xx(1), yy(1), zz(1), 'o', 'Color', colors(uav+1,:), ...
              'MarkerSize',6, 'MarkerFaceColor', colors(uav+1,:));
    end
    % Plot anchors
    scatter3(anchors(:,1), anchors(:,2), anchors(:,3), 80, 'r', '^', 'filled');
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(['(' char('a'+k-1) ') ' traj(k).name]);
    view(35, 25);
    grid on; box on;
    axis equal;
end
sgtitle('Figure 5: Four UAV Swarm Trajectory Types with ISAC Ground Anchors (\blacktriangle)');
saveas(gcf, 'figures\Figure5_trajectories.png');
fprintf('Figure 5 saved.\n');
