data = readtable("rc_0.csv");

time = data{:,1} / 1000;     
velocity = data{:,20};

plot(time, velocity);
xlabel("Time (s)");
ylabel("Air Speed (m/s)");
title("CVT Air Speed");
grid on;

avg_velocity = mean(velocity, "omitnan");
max_velocity = max(velocity, [], "omitnan");
min_velocity = min(velocity, [], "omitnan");

fprintf("Average Velocity: %.2f m/s\n", avg_velocity);
fprintf("Max Velocity: %.2f m/s\n", max_velocity);
fprintf("Min Velocity: %.2f m/s\n", min_velocity);