#ifndef FP_H
#define FP_H

#include "NN_model.h"
#include "math.h"

/*
	@brief forward propagation buffers
	
	For each model's layer, a buffer is defined. Each
	buffer dimension equals to the layer's size.
 */
 typedef struct {
	float layer_0_out[LAYER_0_DIM];
	float layer_1_out[LAYER_1_DIM];
	float layer_2_out[LAYER_2_DIM];
	//float layer_3_out[LAYER_3_DIM];
} forward_propagation_buffers;

/*
	@brief activaiton functions enumeration
	
	Supported activation functions enumeration
 */
 
typedef enum {
	RELU 	= 0,
	SIGMOID = 1,
	TANH 	= 2
} activation_function_type;


/*
	@brief	float neural network forward pass
	@param[in]		input_data		pointer to input data
	@param[in]		model			ML model
	@param[out]		buffers			output buffers
	
	@details
	
	Forward pass function produces the output of each layer
	for a given input. Output buffers size matches the number
	of layer of the ML model. Each buffer size equals each
	layer number of nodes.
*/

void forward_pass(
	float* input_data, 
	ANN_model* model, 
	forward_propagation_buffers* buffers
);

/*
	@brief	float neural network activation function
	@param[in]		node_net_value		pointer to input data
	@param[in]		act_func			ML model
	@return		The function returns the activation operation result
	
	@details
	
	The activation function returns the result of the activion
	function for a given input. Only the folowing activation
	functions are supported:
		(1)	ReLU
		(2)	Sigmoid
		(3)	TanH
*/

float node_activation_function(
	const float node_net_value,
	activation_function_type act_func
);

#endif
