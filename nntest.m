
addpath('./NN');
addpath('./mnistHelper');
addpath('./data');

load_data;

epochs = 1000;
batchsize = 250;
learning_rate = 1e-3;

nn = Network(batchsize);

smooth_loss = log(10);

% ~ 98 % correct
nn.layers{1} = Linear(28 * 28, 256, batchsize);
nn.layers{2} = ReLU(256, 256, batchsize);
nn.layers{3} = Linear(256, 10, batchsize);
nn.layers{4} = Softmax(10, 10, batchsize);

for e=1:1:epochs

    fprintf('Epoch %d...\n', e);
    
    nn.train(train_images, train_labels, learning_rate, size(train_images, 1)/batchsize);
    
    mean(nn.loss)

    smooth_loss = [smooth_loss smooth_loss(end) * 0.99 + 0.01 * nn.loss/batchsize];
    
    figure(1);
    plot(smooth_loss);
    drawnow

end

nn.test(test_images, test_labels);



    