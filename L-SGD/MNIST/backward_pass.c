#include "backward_pass.h"

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

void back_prop(float* input_sample, float* out_trg,	ANN_model* model, forward_propagation_buffers* fwd_p, backward_propagation_buffers* bwd_p, float learning_rate)
{
    //forward_propagation_buffers* fwd_p = (forward_propagation_buffers*)malloc(sizeof(forward_propagation_buffers));
	//backward_propagation_buffers* bwd_p = (backward_propagation_buffers*)malloc(sizeof(backward_propagation_buffers));

    // Forward propagation
	forward_pass(input_sample, model, fwd_p);

	float* curr_layer_wt = model->dense0_wt;
	float* curr_layer_bs = model->dense0_bias;
	float* curr_layer_out = fwd_p->layer_0_out;
	float* prev_layer_out;
	float* next_layer_wt;
	float* next_layer_bs;

	int index;

	//point to last layer
	for (index = 0; index < NUMBER_OF_LAYERS-1; index++)
	{
		curr_layer_wt += model->model_dimensions[index] * model->model_dimensions[index + 1];
		curr_layer_bs += model->model_dimensions[index + 1];
		curr_layer_out += model->model_dimensions[index + 1];
	}

	prev_layer_out = curr_layer_out - model->model_dimensions[index];

	//calculate layer delta values
	float* in_buff = bwd_p->delta1;
	float* out_buff = bwd_p->delta2;
	float* aux_ptr;

	uint8_t* wt_int_width;
	uint8_t* bs_int_width;

    outputLayer_delta(out_trg, curr_layer_out, model->activation_functions[index], model->model_dimensions[index + 1], out_buff);

	for (index = index - 1; index >= 0; index--) //for each layer
	{
		//swap delta buffers
		aux_ptr = in_buff;
		in_buff = out_buff;
		out_buff = aux_ptr;

		//point to layer weights
		next_layer_wt = curr_layer_wt;
		next_layer_bs = curr_layer_bs;
		curr_layer_out = prev_layer_out;
		prev_layer_out = curr_layer_out - model->model_dimensions[index];

		curr_layer_wt -= model->model_dimensions[index] * model->model_dimensions[index + 1];
		curr_layer_bs -= model->model_dimensions[index + 1];
		//uint8_t prev_layer_dim = model->model_dimensions[index];	//uint8_t prev_layer_dim = INPUT_DIM;

        hiddenLayer_delta(in_buff, next_layer_wt, model->model_dimensions[index + 2], curr_layer_out, model->activation_functions[index], model->model_dimensions[index + 1], out_buff);

        update_parameters(in_buff, next_layer_wt, next_layer_bs, curr_layer_out, model->model_dimensions[index + 1], model->model_dimensions[index + 2], learning_rate);

	}

	prev_layer_out = input_sample;

    update_parameters(out_buff, curr_layer_wt, curr_layer_bs, prev_layer_out, model->model_dimensions[0], model->model_dimensions[1], learning_rate);

	//free(fwd_p);
	//free(bwd_p);
}


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

float node_loss_pd(loss_functions loss,	float output_trg, float model_out)
{
	float loss_derivative = 0;
	float num, den;

	switch (loss)
	{
	case MSE:
		loss_derivative = model_out - output_trg;
		break;

	case BCE:
		if (model_out == 0.0)
		{
			num = 1 - output_trg;
			den = 1 - model_out;
		}

		else if (model_out == 1.0)
		{
			num = output_trg;
			den = model_out;
		}

		else
		{
			num = model_out - output_trg;
			den = model_out * (1 - model_out);
		}
		
		loss_derivative = num / den;
		break;

	case CE:
		if (model_out == 0.0)
			model_out = 0.0001;
		loss_derivative = output_trg / model_out;
		break;
	}

	return loss_derivative;

}


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
float node_activation_pd(const float node_output, activation_function_type act_func)
{
	float activation_pd = 0;

	switch (act_func)
	{
	case RELU:
		if (node_output > 0)
			activation_pd = 1.0;
		else
			activation_pd = 0.0;
		break;

	case SIGMOID:
		activation_pd = node_output * (1.0 - node_output);
		break;

	case TANH:
		activation_pd = 1.0 - (node_output * node_output);
		break;

	default:
		break;
	}
	return activation_pd;
}


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

void outputLayer_delta(float* out_target, const float* layer_outs, activation_function_type act_func, const uint8_t layer_size,	float* delta_outBuff)
{
	float delta_value;
	float output_value;
	float output_target = *out_target;
	

#if OUTPUT_FEATURES == 1
	for (int i = 0; i < layer_size; i++)
	{
		delta_value = 0;
		output_value = *(layer_outs + i);
		output_target = *(out_target + i);
		delta_value = node_loss_pd(BCE, output_target, output_value);
		delta_value = delta_value * node_activation_pd(output_value, act_func);

		delta_outBuff[i] = delta_value;
	}

#else
	float targets[OUTPUT_FEATURES] = { 0 };
	targets[(int)output_target] = 1.0;

	for (int i = 0; i < layer_size; i++)
	{
		delta_value = 0;
		output_value = *(layer_outs + i);
		delta_value = node_loss_pd(BCE, *(targets + i), output_value);
		delta_value = delta_value * node_activation_pd(output_value, act_func);

		delta_outBuff[i] = delta_value;
	}
#endif
}


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

void hiddenLayer_delta(float* delta_inBuff,	const float* weights, const uint8_t nextLayer_dim, const float* currLayer_out, activation_function_type act_func, const uint8_t currLayer_dim, float* delta_outBuff)
{
	float new_delta_value = 0;

	float nl_delta;
	float weight;
	float new_delta;
	float output_value;

	for (int node_idx = 0; node_idx < currLayer_dim; node_idx++)
	{
		for (int con_idx = 0; con_idx < nextLayer_dim; con_idx++)
		{
			nl_delta = *(delta_inBuff + con_idx);
			weight = *(weights + con_idx * currLayer_dim + node_idx);
			new_delta = nl_delta * weight;
			new_delta_value += nl_delta * weight;
		}
		output_value = *(currLayer_out + node_idx);
		new_delta_value = new_delta_value * node_activation_pd(output_value, act_func);
		delta_outBuff[node_idx] = new_delta_value;
		new_delta_value = 0;
	}
}

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

void update_parameters(float* delta_inBuff, float* layer_weights, float* layer_bias, float* prevLayer_out, const uint8_t prevLayer_dim, const uint8_t currLayer_dim, float learning_rate)
{
	// memory allocation to store updated parameters
	//float* updated_weights;
	//float* updated_bias;
	float previous_layer_out, weight, bias, learning_update;
	//updated_weights = (float*)malloc(prevLayer_dim * currLayer_dim * sizeof(float));
	//updated_bias = (float*)malloc(currLayer_dim * sizeof(float));

	float node_delta;
	float* wt_ptr;
	float* prev_layer_out_ptr;

	for (int node_idx = 0; node_idx < currLayer_dim; node_idx++) {

		node_delta = *(delta_inBuff + node_idx);

		for (int out_idx = 0; out_idx < prevLayer_dim; out_idx++)
		{
			wt_ptr = layer_weights + prevLayer_dim * node_idx + out_idx;
			prev_layer_out_ptr = prevLayer_out + out_idx;

			//	compute new weight
			previous_layer_out = *(prev_layer_out_ptr);
			weight = *(wt_ptr);
			learning_update = previous_layer_out * node_delta * learning_rate;
			//*(updated_weights + prevLayer_dim * node_idx + out_idx) = weight - learning_update;
			*(wt_ptr) = weight - learning_update;
		}

		//	compute new bias
		bias = *(layer_bias + node_idx);
		learning_update = learning_rate * node_delta;
		//*(updated_bias + node_idx) = bias - learning_update;
		*(layer_bias + node_idx) = bias - learning_update;
	}

	//int conn_size = currLayer_dim * prevLayer_dim;
	//for (int wt_idx = 0; wt_idx < conn_size; wt_idx++)
	//{
	//	float* wt_ptr = layer_weights + wt_idx;
	//	*wt_ptr = *(updated_weights + wt_idx);
	//}
	//
	//for (int bs_idx = 0; bs_idx < currLayer_dim; bs_idx++)
	//{
	//	float* bs_ptr = layer_bias + bs_idx;
	//	*bs_ptr = *(updated_bias + bs_idx);
	//}

	//free(updated_weights);
	//free(updated_bias);

}   
