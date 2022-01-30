#ifndef NN_PD_H
#define NN_PD_H
#include "stdint.h"

#define MODEL	MNIST_MODEL
#define PD_MODEL	1
#define MNIST_MODEL	2

#if MODEL == PD_MODEL
	#define NUMBER_OF_LAYERS		3
	#define INPUT_FEATURES 			6  
	#define OUTPUT_FEATURES 		1
	#define LAYER_0_DIM 			40
	#define LAYER_1_DIM 			32
	#define LAYER_2_DIM 			OUTPUT_FEATURES
	#define LAYERS_DIMS				{ INPUT_FEATURES, LAYER_0_DIM, LAYER_1_DIM, OUTPUT_FEATURES }				
	#define ACTIVATION_FUNCS		{ TANH, TANH, SIGMOID }									

	#define L0_WT_DIM INPUT_FEATURES*LAYER_0_DIM
	#define L1_WT_DIM LAYER_0_DIM*LAYER_1_DIM
	#define L2_WT_DIM LAYER_1_DIM*LAYER_2_DIM

	typedef struct {
		float dense0_wt[INPUT_FEATURES * LAYER_0_DIM];
		float dense1_wt[L1_WT_DIM];
		float dense2_wt[L2_WT_DIM];
		// dense3_wt[L3_WT_DIM];

		float dense0_bias[LAYER_0_DIM];
		float dense1_bias[LAYER_1_DIM];
		float dense2_bias[LAYER_2_DIM];
		//float dense3_bias[LAYER_3_DIM];

		int model_dimensions[NUMBER_OF_LAYERS + 1];
		int activation_functions[NUMBER_OF_LAYERS];

	} ANN_model;

#elif MODEL == MNIST_MODEL
	#define NUMBER_OF_LAYERS		3
	#define INPUT_FEATURES			784      
	#define OUTPUT_FEATURES 		10
	#define LAYER_0_DIM 			40
	#define LAYER_1_DIM 			32
	#define LAYER_2_DIM 			OUTPUT_FEATURES
	#define LAYERS_DIMS				{ INPUT_FEATURES, LAYER_0_DIM, LAYER_1_DIM, OUTPUT_FEATURES }				
	#define ACTIVATION_FUNCS		{ TANH, TANH, SIGMOID }									

	#define L0_WT_DIM INPUT_FEATURES*LAYER_0_DIM
	#define L1_WT_DIM LAYER_0_DIM*LAYER_1_DIM
	#define L2_WT_DIM LAYER_1_DIM*OUTPUT_FEATURES

	typedef struct {
		float dense0_wt[INPUT_FEATURES * LAYER_0_DIM];
		float dense1_wt[L1_WT_DIM];
		float dense2_wt[L2_WT_DIM];
		// dense3_wt[L3_WT_DIM];

		float dense0_bias[LAYER_0_DIM];
		float dense1_bias[LAYER_1_DIM];
		float dense2_bias[LAYER_2_DIM];
		//float dense3_bias[LAYER_3_DIM];

		int model_dimensions[NUMBER_OF_LAYERS + 1];
		int activation_functions[NUMBER_OF_LAYERS];
	} ANN_model;
#endif

//#define L3_WT_DIM LAYER_2_DIM*LAYER_3_DIM



#endif