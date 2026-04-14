

for i=1:1:49;
    %figure(gcf)
plot(q_mat,5200*dthat1(i,1:8),'DisplayName','dthat1(1,1:8)','YDataSource','dthat1(1,1:8)');
hold on;
plot(q_mat,5200*Y(i,1:8),'r','DisplayName','Y(1,1:8)','YDataSource','Y(1,1:8)');
figure%(gcf)
end