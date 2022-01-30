#ifndef BP_H
#define BP_H

#include "forward_pass.h"
#include "stdlib.h"

#define MAX_LAYER_SIZE	40
#define Q15_FRAC_BITS	8
#define Q15_SHIFT		Q15_FRAC_BITS-7


/*
	@brief loss functions

	Supported loss functions enumeration
 */

typedef enum {
	MSE = 0,
	BCE = 1,
	CE = 2
} loss_functions;


/*
	@brief backward propagation buffers
	
	There are two buffers define witth the MAX_LAYER_SIZE size
	to store delta values during training process
 */

typedef struct {
	float delta1[MAX_LAYER_SIZE];
	float delta2[MAX_LAYER_SIZE];
}backward_propagation_buffers;

/*
	@brief	neural network backward propagation
	@param[in]			input_sample	forward pass input sample
	@param[in]			model			ML model
	@param[in, out]		learning_rate	training learning rate	
	@param[in]			out_trg			output target value
	
	@details
	
	Backpropagation function updates ML model parameters based
	on the backpropagtion process. Thus, a input value is fed
	to the model and the produced output is compared with the
	output target (label).
*/

void back_prop(
	float* input_sample,
	float* out_trg,
	ANN_model* model,
	forward_propagation_buffers* fwd_p, 
	backward_propagation_buffers* bwd_p,
	float learning_rate
);

/*
	@brief	loss partial derivative computation
	@param[in]			loss			loss function
	@param[in]			output_trg		sample label
	@param[in]			model_out		model prediction	
	@return		The function returns the loss function partial 
				derivative result
	
	@details
	
	This function computes the loss function partial derivative 
	of a given node. This computation relies on the model's 
	activation function, the model's output and the sample's label.
*/

float node_loss_pd(
	loss_functions loss,
	float output_trg,
	float model_out
);


/*
	@brief	activation partial derivative computation
	@param[in]			act_func		activation function
	@param[in]			output_trg		sample label
	@return		The function returns the activation partial 
				derivative result
	
	@details
	
	This function computes the activation function partial 
	derivative of a given node. This computation relies on 
	the model's activation function, the model's output and 
	the sample's label.
*/

float node_activation_pd(
	const float node_output,
	activation_function_type act_func
);

/*
	@brief	output layer delta computation
	@param[in]			out_target		labels
	@param[in]			layer_outs		model predictions
	@param[in]			act_func		activation function
	@param[in]			layer_size		layer dimension
	@param[in, out]		delta_outBuff	output deltas values buffer
	
	@details
	
	The output layer delta values depend  on the partial
	derivative of the loss function and the partial derivative
	of the activation function. The output buffer contains
	the delta values of each layer's node. Thus, the number
	of computed deltas equals to the layer size.
*/

void outputLayer_delta(
	float* out_target,
	const float* layer_outs,
	activation_function_type act_func,
	const uint8_t layer_size,
	float* delta_outBuff
);


/*
	@brief	hidden layer delta computation
	@param[in]			delta_inBuff	input deltas values buffer
	@param[in]			weights			next layer weigths buffer
	@param[in]			nextLayer_dim	next layer dimension
	@param[in]			currLayer_out	current layer outputs buffer
	@param[in]			act_func		activation function
	@param[in, out]		delta_outBuff	output deltas values buffer
	
	@details

	The hidden lauer delta values depend on the previous computed
	deltas and the weights that compose the connection between
	layers (next layer weights). The output buffer containt the
	delta values of each current layer's nodes. Thus, the number
	of computed deltas equals current layer size.
*/

void hiddenLayer_delta(
	float* delta_inBuff,
	const float* weights,
	const uint8_t nextLayer_dim,
	const float* currLayer_out,
	activation_function_type act_func,
	const uint8_t currLayer_dim,
	float* delta_outBuff
);


/*
	@brief	update layer's parameters
	@param[in]			delta_inBuff	input deltas values buffer
	@param[in, out]		weights			layer weigths buffer
	@param[in, out]		bias			layer bias buffer
	@param[in]			prevLayer_out	previous layer outputs buffer
	@param[in]			prevLayer_dim	previous layer dimension
	@param[in]			currLayer_dim	current layer dimension
	@param[in]			learning_rate	training learning rate
	
	@details

	The update parameters functions update a given layer parameters
	(weigths and bias) based on the delta values buffers, the previous
	layer outputs and the learning rate.
*/

void update_parameters(
	float* delta_inBuff,
	float* weights,
	float* bias,
	float* prevLayer_out,
	const uint8_t prevLayer_dim,
	const uint8_t currLayer_dim,
	float learning_rate
);

#endif