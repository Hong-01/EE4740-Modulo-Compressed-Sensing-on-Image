% Modulo compressed sensing

% Test for MILP and OMP with different M
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
%% Import the data


data = readmatrix("E:\DATA\TUD\Master\TUD_Master_Y1\Q3\EE4740 Data Compression Entropy and Sparsity Perspectives\Final Project\Data\mnist_test.csv");
data = data(1,2:end);
nnz(data)
n = length(data)
img_size=sqrt(n);

figure;
imshow(reshape(data,[img_size,img_size]),[],'InitialMagnification', 'fit');


%% Image preprocessing

% threshold procress
threshold = 200;
data_comp = reshape(data,[1,n]);
data_comp(data_comp < threshold) = 0;
data_comp(data_comp >= threshold) = 1; %normalize 255 to 1 to boost the process speed
figure;
imshow(reshape(data_comp,[img_size,img_size]),[],'InitialMagnification', 'fit');
title("Image after threshold")


s=nnz(data_comp);      %calculate the sparsity

m_list=125:25:600      %Define the list of M


%% Calculate omp and milp
time_milp_list=[];
time_omp_list=[];
x_rec_milp_list=[];
x_rec_omp_list=[];

for i = 1:length(m_list)
   
    m=m_list(i)
    variance = 1/m;
    A =randn(m, n)*sqrt(variance);
    [x_rec_milp,time_milp]=MILP(m,n,data_comp,A);
    [x_rec_omp,time_omp]=omp(m,n,data_comp,s,A);

    time_milp_list=[time_milp_list,time_milp];
    time_omp_list=[time_omp_list,time_omp];

    x_rec_milp_list=[x_rec_milp_list;x_rec_milp];
    x_rec_omp_list=[x_rec_omp_list;x_rec_omp];



end

%% plot result


% plot recovered images
num_images = size(x_rec_milp_list, 1);
num_rows = ceil(sqrt(num_images));
num_cols = ceil(num_images / num_rows);

figure;
for i = 1:num_images
    subplot(num_rows, num_cols, i);
    imshow(reshape(x_rec_milp_list(i, :), [img_size, img_size]), [],'InitialMagnification', 'fit'); 
    title(['M = ', num2str(m_list(i))]);
end
sgtitle('Reconstructed Images with MILP');

figure;
for i = 1:num_images
    subplot(num_rows, num_cols, i);
    imshow(reshape(x_rec_omp_list(i, :), [img_size, img_size]), [],'InitialMagnification', 'fit'); 
    title(['M = ', num2str(m_list(i))]);
end
sgtitle('Reconstructed Images with OMP');

% plot computational time
figure;
plot(m_list, time_milp_list,'o-')
xlabel('M');
ylabel('seconds');
title('MILP-Computional time with diffrent M');

figure;
plot(m_list, time_omp_list,'o-')
xlabel('M');
ylabel('seconds');
title('OMP-Computional time with diffrent M');

% calculate milp mse
mse_milp_list=[];
for i = 1 : num_images
    mse_value=log10(sum((data_comp-x_rec_milp_list(i, :)).^2));
    mse_milp_list=[mse_milp_list,mse_value];
end

% calculate omp mse
mse_omp_list=[];                            
for i = 1 : num_images
    mse_value=log10(sum((data_comp-x_rec_omp_list(i, :)).^2));
    mse_omp_list=[mse_omp_list,mse_value];
end

%plot mse
figure;
hold on;
plot(m_list, mse_milp_list,'o-')
plot(m_list, mse_omp_list,'s-')
hold off;
xlabel('M');
ylabel('mse (dB)');
title('MSE');
legend('MILP', 'OMP');













