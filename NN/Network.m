%
% @Author: kmrocki
% @Date:   2016-12-09 12:01:23
% @Last Modified by:   kmrocki
% @Last Modified time: 2016-12-09 12:01:23
% 

classdef Network < handle

	properties

		layers;
		batchsize;
		loss;

	end

	methods

		function obj = Network(batchsize)

			obj.batchsize = batchsize;
			obj.layers = [];

		end

		function obj = forward(obj, input_data)

			obj.layers{1}.x = input_data';

			% go forward
			for i=1:1:length(obj.layers)

				obj.layers{i}.forward();

				if (i < length(obj.layers))
					
					obj.layers{i+1}.x = obj.layers{i}.y;

				end
			
			end
		
		end

		function obj = backward(obj, targets)

			I = eye(10);

			obj.layers{end}.dy = I(:, targets+1);

			% go back
			for i=length(obj.layers):-1:1

				obj.layers{i}.reset_grads();
				obj.layers{i}.backward();

				if (i > 1)
					obj.layers{i-1}.dy = obj.layers{i}.dx;
				end

			end
		end

			function obj = update(obj, alpha)

				for i=1:1:length(obj.layers)

					obj.layers{i}.apply_grads(alpha);

				end

		end

		function obj = train(obj, train_images, train_labels, learning_rate, iterations)

			num_examples = size(train_images, 1);

			for ii=1:1:iterations

				random_numbers = randi([1 num_examples], 1, obj.batchsize);

				batch = train_images(random_numbers, :);
				targets = train_labels(random_numbers, :);

				% forward activations
				obj.forward(batch);

				I = eye(10);

					targets_onehot = I(:, targets+1);

				obj.loss = cross_entropy(obj.layers{end}.y, targets_onehot);
				%fprintf('[ \t %d / %d\t ] Loss = %f\n', ii, iterations, loss);

					% % backprogagation
					obj.backward(targets);

					% % apply changes
					obj.update(learning_rate);

			end

		end

		function obj = test(obj, test_images, test_labels)

			loss = 0;
			num_correct = 0;
			num_cases = size(test_images, 1);

			for ii=1:1:num_cases/obj.batchsize

				numbers = [(ii-1)*obj.batchsize+1:1:ii*obj.batchsize];

				batch = test_images(numbers, :);
				targets = test_labels(numbers, :);

				% forward activations
				obj.forward(batch);

				[output idx] = max(obj.layers{end}.y, [], 1);

				num_correct = num_correct + sum(targets+1 == idx');

			end

			fprintf('%% correct = %.2f\n', 100.0 * num_correct/num_cases);

		end

	end

end
