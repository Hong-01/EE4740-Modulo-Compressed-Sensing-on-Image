% Modulo compressed sensing
% Test MILP by different M,S
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
%% Import the data
image_size=28;
n=image_size*image_size;
data = readmatrix("E:\DATA\TUD\Master\TUD_Master_Y1\Q3\EE4740 Data Compression Entropy and Sparsity Perspectives\Final Project\Data\mnist_test.csv");
data = data(1,2:end);

%% Image preprocessing
%change sparsity by threshold
th_val_list = [130,160,186,205,220,228,241,251];
th_img_list = cell(1, length(th_val_list));
for i = 1:length(th_val_list)
    data_process=data;
    data_process(data_process < th_val_list(i)) = 0;
    data_process(data_process >= th_val_list(i)) = 1;
    th_img_list{i} = data_process;
end

%calculate sparse
sparse_values = zeros(1, length(th_val_list));
for i = 1:length(th_val_list)
    current_image = th_img_list{i};
    sparse_values(i)  = nnz(current_image);
end

% display S
disp('Sparse values for each quantized image:');
disp(sparse_values);

% Show processed image
figure;
for i = 1:length(th_val_list)
    subplot(2,4, i);
    image=reshape(th_img_list{i},[28,28]);
    imshow(image,[]);
    title(['Sparsity: ', num2str(sparse_values(i))]);
end


m_list=[200,250,300,350,400,450,500,550]



%% Running in MILP and OMP

mse_list=[];
time_list=[];

for j=1:length(m_list)
    m=m_list(j)
    mse_record=[];
    time_record=[];
    for i = 1:length(th_img_list)
        %MILP(m,n,th_img_list{0})
        s=sparse_values(i)
        input_data=double(th_img_list{i});
        [x_rec,time]=MILP(m,n,input_data);
        mse_value=log10(sum((double(th_img_list{i})-x_rec).^2));
        mse_record=[mse_record,mse_value];
        time_record=[time_record,time];
    end

    mse_list=[mse_list;mse_record];
    time_list=[time_list;time_record];
end

%% Evaluation
% plot mse
figure;
hold on;
dot_shapes = {'o-', 's-', 'd-', '^-', 'p-', 'h-','*-','V-'};
for i= 1:length(m_list)
    plot(sparse_values, mse_list(i,:),dot_shapes{i}, 'DisplayName', ['M=', num2str(m_list(i))])
end
legend('show')
xlabel('S');
ylabel('mse (dB)');
title('MILP MSE');
hold off;

% plot computional time
figure;
hold on;
dot_shapes = {'o-', 's-', 'd-', '^-', 'p-', 'h-','+-','s-'};
for i = 1:length(m_list)
    plot(sparse_values, time_list(i,:), dot_shapes{i}, 'DisplayName', ['M=', num2str(m_list(i))])
end
legend('show')
xlabel('S');
ylabel('seconds');
title('MILP computational time');
hold off;













