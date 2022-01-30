#ifndef TP_H
#define TP_H

#include "backward_pass.h"
#include <stdio.h>

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

void training_process(ANN_model* model, forward_propagation_buffers* fwd_p, backward_propagation_buffers* bwd_p,
	int train_set_size,
	int test_set_size,
	float* training_samples,
	float* training_labels,
	float* validation_samples,
	float* validation_labels,
	uint8_t epochs,
	float learning_rate
);


/*
	@brief	neural network validation process
	@param[in]			model					ML model
	@param[in]			values					validation subset values
	@param[in]			labels					validation subset labels
	@param[in]			data_size				number of validation samples
	@return		The function returns the validation accuracy
*/

float validation_process(
	ANN_model* model,
	forward_propagation_buffers* fwd_p,
	float* values,
	float* labels,
	int data_size
);


/*
	@brief	data shuffle process
	@param[in, out]		data_values				values to be shuffled
	@param[in, out]		data_labels				labels to be shuffled
	@param[in]			samples_dim				number of samples
	@param[in]			input_dim				batch size of each sample

	
	@details
	
	Shuffling process based on modern Fisher-Yates algorithm
*/

void shuffle_data(
	float* data_values,
	float* data_labels,
	const int samples_dim,
	const int input_dim
);

#endif