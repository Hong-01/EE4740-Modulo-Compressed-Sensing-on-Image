% Modulo compressed sensing
% Test for MILP and OMP with different quantization bits L
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
%% Import the data
% img_size=28;

big_image_size=28;
n=big_image_size*big_image_size;
data = readmatrix("E:\DATA\TUD\Master\TUD_Master_Y1\Q3\EE4740 Data Compression Entropy and Sparsity Perspectives\Final Project\Data\mnist_test.csv");
data = data(1,2:end);

%% Image process

%interpolation to get smooth gray scale
data=reshape(data,[28,28]);
data_big=reshape(data,[28,28]);
data_big=imresize(data_big, [256,256], 'bilinear'); 
data_big=imresize(data_big, [28,28]); 

figure; %with interpolation
imshow(data_big, [],'InitialMagnification', 'fit');

figure; %original
imshow(data, [],'InitialMagnification', 'fit')

n = length(reshape(data_big,[big_image_size*big_image_size,1]));
img_size=sqrt(n);

%quantization
quan_val_list = [1, 2, 3,4];
quan_img_list = cell(1, length(quan_val_list));
for i = 1:length(quan_val_list)
    quantization_bit = quan_val_list(i); 
    quantization_level = 2^quantization_bit;
    quantized_image = round((data_big/255) * (quantization_level-1)) * (255/(quantization_level-1));
    quantized_image = double(reshape(quantized_image, [big_image_size, big_image_size])); 
    quan_img_list{i} = quantized_image;
end

% Display of the quantized image
figure;
for i = 1:length(quan_val_list)
    subplot(1,4, i);
    imshow(quan_img_list{i},[]);
    title(['Quantized Bits: ', num2str(quan_val_list(i))]);
end


%calculate sparse
sparse_values = zeros(1, length(quan_val_list));
for i = 1:length(quan_val_list)
    current_image = quan_img_list{i};
    sparse_values(i)  = nnz(current_image);
end

% show sparsity
disp('Sparse values for each quantized image:');
disp(sparse_values);

%we can find S is different in different images


%reshape to vector
for i=1:length(quan_img_list)
    quan_img_list{i}=reshape(quan_img_list{i},[1,n]);
    for k=1:length(quan_img_list{i})
        quan_img_list{i}(k)=quan_img_list{i}(k)/255;
    end
end

% convert all sparsity is same as the lowest sparse one
zero_indices = find(quan_img_list{1} == 0);
for i=2:length(quan_img_list)
    quan_img_list{i}(zero_indices)=0;
end

figure; %show the image after preprocess
for i = 1:length(quan_val_list)
    subplot(1,4, i);
    image=reshape(quan_img_list{i},[28,28]);
    imshow(image,[]);
    title(['Quantized Bits: ', num2str(quan_val_list(i))]);
end



%calculate sparse again to check whether S is same
sparse_values = zeros(1, length(quan_val_list));
for i = 1:length(quan_val_list)
    current_image = quan_img_list{i};
    sparse_values(i)  = nnz(current_image);
end

% show S
disp('Sparse values for each quantized image:');
disp(sparse_values);

s=mean(sparse_values)      %calculate the sparsity
m = 260   % measurement number

if m>=n
    disp("ERROR: NOT SPARSE ENOUGH, m >= n")
end

%% Calculate omp and milp
time_milp_list=[];
time_omp_list=[];
x_rec_milp_list=[];
x_rec_omp_list=[];
variance = 1/m;
A =randn(m, n)*sqrt(variance);
for i = 1:length(quan_val_list)
    
    L = quan_val_list(i)
    [x_rec_milp,time_milp]=MILP(m,n,quan_img_list{i},A);
    [x_rec_omp,time_omp]=omp(m,n,quan_img_list{i},sparse_values(i),A);

    time_milp_list=[time_milp_list,time_milp];
    time_omp_list=[time_omp_list,time_omp];

    x_rec_milp_list=[x_rec_milp_list;x_rec_milp];
    x_rec_omp_list=[x_rec_omp_list;x_rec_omp];
end
%% plot result image

figure;
for i = 1:length(quan_img_list)
    subplot(1,4, i);
    imshow(reshape(x_rec_milp_list(i, :), [img_size, img_size]), [],'InitialMagnification', 'fit'); 
    title(['L = ', num2str(quan_val_list(i))]);
end
sgtitle('Reconstructed Images with MILP');

figure;
for i = 1:length(quan_img_list)
    subplot(1,4, i);
    imshow(reshape(x_rec_omp_list(i, :), [img_size, img_size]), [],'InitialMagnification', 'fit'); 
    title(['L = ', num2str(quan_val_list(i))]);
end
sgtitle('Reconstructed Images with OMP');
%% plot evaluation charts

% plot computational time
figure;
plot(quan_val_list, time_milp_list,'o-')
xlabel('L');
ylabel('second');
title('MILP-Computional time with diffrent L');

% plot time list
figure;
plot(quan_val_list, time_omp_list,'o-')
xlabel('L');
ylabel('second');
title('OMP-Computional time with diffrent L');

% calculate mse
mse_milp_list=[];
for i = 1 : 1:length(quan_img_list)
    % mse_value=mse(data_comp, x_rec);
    mse_value=log10(sum((quan_img_list{i}-x_rec_milp_list(i, :)).^2));
    mse_milp_list=[mse_milp_list,mse_value];
end

mse_omp_list=[];                            
for i = 1 : 1:length(quan_img_list)
    % mse_value=mse(data_comp, x_rec);
    mse_value=log10(sum((quan_img_list{i}-x_rec_omp_list(i, :)).^2));
    mse_omp_list=[mse_omp_list,mse_value];
end

%plot mse
figure;
hold on;
plot(quan_val_list, mse_milp_list,'o-')
plot(quan_val_list, mse_omp_list,'s-')
hold off;
xlabel('L');
ylabel('mse (dB)');
title('MSE');
legend('MILP', 'OMP');








