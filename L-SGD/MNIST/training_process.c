#include "training_process.h"

/*
/*
	@brief	neural network training process
	@param[in, out]		model					ML model
	@param[in]			train_set_size			number of train samples
	@param[in]			test_set_size			number of validation samples
	@param[in]			training_samples		training subset values
	@param[in]			training_labels			training subset labels
	@param[in]			validation_samples		validation subset values
	@param[in]			validation_labels		validation subset labels
	@param[in]			epochs					number of epochs
	@param[in]			learning_rate			training learning rate	
	
    @details
	
	Training process functions performs a traing over the input model
	with the defined hyperparameters (epochs and learning rate) and 
	with the dataset (subset of training and subset of validation).
	The training process only proceeds with unitary batch size due to
	memory constraints.
*/

void training_process(ANN_model* model, forward_propagation_buffers* fwd_p, backward_propagation_buffers* bwd_p, int train_set_size, int test_set_size,
					float* training_samples, float* training_labels, float* validation_samples, float* validation_labels, uint8_t epochs, float learning_rate)
{
	float train_acc, val_acc;

	//forward_propagation_buffers* fwd_p = (forward_propagation_buffers*)malloc(sizeof(forward_propagation_buffers));
	//backward_propagation_buffers* bwd_p = (backward_propagation_buffers*)malloc(sizeof(backward_propagation_buffers));


	for (int curr_epoch = 1; curr_epoch <= epochs; curr_epoch++)             //run n epochs
	{
		//	Shuffle data
		//shuffle_data(training_samples, training_labels, train_set_size, INPUT_DIM);	//train data shuffle
		//shuffle_data(validation_samples, validation_labels, test_set_size, INPUT_DIM);	//validation data shuffle

		//	Training
		float* sample = training_samples;
		float* label = training_labels;
		for (int sample_index = 0; sample_index < train_set_size; sample_index++)         //iterate input samples
		{
			back_prop(sample, label, model, fwd_p, bwd_p, learning_rate);
			label++;
			sample += INPUT_FEATURES;
		}
		
		//train_acc = validation_process(model, fwd_p, training_samples, training_labels, train_set_size);
		val_acc = validation_process(model, fwd_p, validation_samples, validation_labels, test_set_size);
		//printf("Epoch %d/%d:\t\ttraining accuracy: %f\t\tvalidation accuracy: %f\n", curr_epoch, epochs, train_acc, val_acc);
		printf("Epoch %d/%d:\t\tvalidation accuracy: %f\n", curr_epoch, epochs, val_acc);
	}

	//free(fwd_p);
	//free(bwd_p);
}


/*
	@brief	neural network validation process
	@param[in]			model					ML model
	@param[in]			values					validation subset values
	@param[in]			samples					validation subset labels
	@param[in]			data_size				number of validation samples
	@return		The function returns the validation accuracy
*/

float validation_process(ANN_model* model, forward_propagation_buffers* fwd_p, float* values, float* labels, int data_size)
{
	int correct_predictions = 0;
	float* input_sample	= values;
	float* output_label	= labels;

	//forward_propagation_buffers* fwd_p = (forward_propagation_buffers*)malloc(sizeof(forward_propagation_buffers));

	for (int sample_index = 0; sample_index < data_size; sample_index++)
	{
		forward_pass(input_sample, model, fwd_p);

		//if (fwd_p->layer_2_out[0] > 0.45 && *output_label == 1)
		//	correct_predictions++;
		//
		//else if (fwd_p->layer_2_out[0] <= 0.45 && *output_label == 0)
		//	correct_predictions++;

		//Get most likely output
		int out_dim = model->model_dimensions[3];
		float max_value = 0.0;
		int node_idx;

		if (out_dim > 1)
		{
			node_idx = 0;
			for (int output_idx = 0; output_idx < out_dim; output_idx++)
			{
				float output_value = fwd_p->layer_2_out[output_idx];
				if (max_value < output_value)
				{
					max_value = output_value;
					node_idx = output_idx;
				}
			}
		}

		else if (fwd_p->layer_2_out[0] > 0.45)
		{
			node_idx = 1;
		}

		else
			node_idx = 0;

		if ((int)node_idx == (int)*output_label)
			correct_predictions++;

		input_sample += INPUT_FEATURES;
		output_label++;
	}

	//free(fwd_p);

	return(float)correct_predictions/data_size;
}


/*
	@brief	data shuffle process
	@param[in, out]		data_values				values to be shuffled
	@param[in, out]		data_labels				labels to be shuffled
	@param[in]			samples_dim				number of samples
	@param[in]			input_dim				batch size of each sample

	
	@details
	
	Shuffling process based on modern Fisher-Yates algorithm
*/

void shuffle_data(float* data_values, float* data_labels, const int samples_dim, const int input_dim)
{
	float* shuffle_sample	= (float*)malloc(sizeof(float) * input_dim);

	int rand_index = 0;
	int last_index = samples_dim - 1;

	float* rand_sample_ptr;
	float* last_index_sample_ptr = data_values + (last_index * input_dim) ;

	float* rand_label_ptr;
	float* last_index_label_ptr = data_labels + last_index;

	

	while (last_index != 0)
	{
		srand(time(0));
		rand_index = rand() % last_index;

		//	Swap data values
		rand_sample_ptr = data_values + (rand_index * input_dim);
		memcpy(shuffle_sample, rand_sample_ptr, (input_dim * sizeof(float)));
		memcpy(rand_sample_ptr, last_index_sample_ptr, (input_dim * sizeof(float)));
		memcpy(last_index_sample_ptr, shuffle_sample, (input_dim * sizeof(float)));

		//	Swap labels
		rand_label_ptr = data_labels + rand_index;
		float label_temp = *rand_label_ptr;
		*rand_label_ptr = *last_index_label_ptr;
		*last_index_label_ptr = label_temp;

		//	Decrement index pointers
		last_index--;
		last_index_sample_ptr -= input_dim;
		last_index_label_ptr--;
	};
	
	free(shuffle_sample);
}


