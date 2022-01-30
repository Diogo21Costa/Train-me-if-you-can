#include "stdlib.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "NN_model.h"
//#include "Model2_dataset.h"
#include "training_process.h"


#define TIME_MEASURE	1
#define SETUP			PRIVATE_DATASET	
#define MNIST			1
#define PRIVATE_DATASET	2

#if SETUP == PRIVATE_DATASET
	#define TRAIN_SIZE      4032
	#define TEST_SIZE       1728
	#define EPOCHS          1
	#define LEARNING_RATE   0.01
	#include "train_data_PD.h"
	#include "test_data_PD.h"
	#include "model_parameters_PD.h"

#elif SETUP == MNIST
	#define TRAIN_SIZE      14700
	#define TEST_SIZE       6300
	#define EPOCHS          1
	#define LEARNING_RATE   0.01
	#include "train_data_MNIST.h"
	#include "test_data_MNIST.h"
	#include "model_parameters_MNIST.h"
#endif


#if TIME_MEASURE
	#include "stdlib.h"
	#include <stdint.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include <time.h>
#endif



//#define NUMBER_OF_SAMPLES		3840
#define RUNS	20

const signed int Model_Dimensions[] = LAYERS_DIMS;
const signed int Activation_Functions[] = ACTIVATION_FUNCS;

void load_model_cfg(ANN_model* model)
{
	memcpy(model->dense0_wt, LAYER_0_WT, sizeof(LAYER_0_WT));
	memcpy(model->dense1_wt, LAYER_1_WT, sizeof(LAYER_1_WT));
	memcpy(model->dense2_wt, LAYER_2_WT, sizeof(LAYER_2_WT));
	//memcpy(model->dense3_wt, LAYER_2_WT, sizeof(LAYER_2_WT));

	memcpy(model->dense0_bias, LAYER_0_BS, sizeof(LAYER_0_BS));
	memcpy(model->dense1_bias, LAYER_1_BS, sizeof(LAYER_1_BS));
	memcpy(model->dense2_bias, LAYER_2_BS, sizeof(LAYER_2_BS));
	//memcpy(model->dense3_bias, LAYER_2_BS, sizeof(LAYER_2_BS));

	memcpy(model->model_dimensions, Model_Dimensions, sizeof(Model_Dimensions));
	memcpy(model->activation_functions, Activation_Functions, sizeof(Activation_Functions));
}

void main()
{
	ANN_model* new_model = (ANN_model*)malloc(sizeof(ANN_model));
	forward_propagation_buffers* fwd_p = (forward_propagation_buffers*)malloc(sizeof(forward_propagation_buffers));
	backward_propagation_buffers* bwd_p = (backward_propagation_buffers*)malloc(sizeof(backward_propagation_buffers));

	load_model_cfg(new_model);


#if TIME_MEASURE == 1
	for (int run = 0; run < RUNS; run++)
	{
		clock_t begin = clock();
		training_process(
			new_model,
			fwd_p,
			bwd_p,
			TRAIN_SIZE,
			TEST_SIZE,
			train_samples,
			train_labels,
			test_samples,
			test_labels,
			EPOCHS,
			LEARNING_RATE
		);
		clock_t end = clock();
		printf("Elapsed: %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
	}

#else
	float accuracy = validation_process(new_model, fwd_p, test_samples, test_labels, TEST_SIZE);
	printf("Initial validation accuracy: %f", accuracy);

	training_process(
		new_model,
		fwd_p,
		bwd_p,
		TRAIN_SIZE,
		TEST_SIZE,
		train_samples,
		train_labels,
		test_samples,
		test_labels,
		EPOCHS,
		LEARNING_RATE
	);
#endif

	int x = 0;
	x = x + 1;
	//new_model->dense0_bias;
	//new_model->dense1_bias;
	//new_model->dense2_bias;
	//
	//new_model->dense0_wt;
	//new_model->dense1_wt;
	//new_model->dense2_wt;
	//
	//FILE* f = fopen("dense0_wt.txt", "w");
	//for (int i=0; i< L0_WT_DIM; i++)
	//{
	//	fprintf(f, "%f, ", new_model->dense0_wt[i]);
	//}
	//fclose(f);
	//
	//f = fopen("dense1_wt.txt", "w");
	//for (int i = 0; i < L1_WT_DIM; i++)
	//{
	//	fprintf(f, "%f, ", new_model->dense1_wt[i]);
	//}
	//fclose(f);
	//
	//f = fopen("dense2_wt.txt", "w");
	//for (int i = 0; i < L2_WT_DIM; i++)
	//{
	//	fprintf(f, "%f, ", new_model->dense2_wt[i]);
	//}
	//fclose(f);
	//
	//
	//f = fopen("dense0_bs.txt", "w");
	//for (int i = 0; i < LAYER_0_DIM; i++)
	//{
	//	fprintf(f, "%f, ", new_model->dense0_bias[i]);
	//}
	//fclose(f);
	//
	//f = fopen("dense1_bs.txt", "w");
	//for (int i = 0; i < LAYER_1_DIM; i++)
	//{
	//	fprintf(f, "%f, ", new_model->dense1_bias[i]);
	//}
	//fclose(f);
	//
	//f = fopen("dense2_bs.txt", "w");
	//for (int i = 0; i < LAYER_2_DIM; i++)
	//{
	//	fprintf(f, "%f, ", new_model->dense2_bias[i]);
	//}
	//fclose(f);
	//
	//int x = 0;

}