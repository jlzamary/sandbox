#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Simple neural network that can learn xor function

// Sigmoid function for back propagation
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

// Take derivative of sigmoid function to update weights in back propagation
double dsigmoid(double x) { return x * (1 - x); }

// Function to initialize weights
// Random number between 0 and 1
double init_weights() { return ((double)rand()) / ((double)RAND_MAX); }

// Shuffle function to randomly shuffle data
void shuffle(int *array, size_t n)
{

    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = 1 + rand() / (RAND_MAX / (n - 1) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

int main(void)
{

    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    // Layer bias
    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    // Add weights between hidden nodes
    // 2D array with hidden weights
    double hiddenWeights[numInputs][numHiddenNodes];

    double outputWeights[numHiddenNodes][numOutputs];

    // 2D training array
    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                          {1.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 1.0f}};

    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};

    // Initialize weights with random values
    for (int i = 0; i < numInputs; i++)
    {
        for (int j = 0; j < numHiddenNodes; j++)
        {
            hiddenWeights[i][j] = init_weights();
        }
    }

    // Output weights
    for (int i = 0; i < numHiddenNodes; i++)
    {
        for (int j = 0; j < numOutputs; j++)
        {
            outputWeights[i][j] = init_weights();
        }
    }

    // Bias
    // Only single as the output is a one dimensional array
    for (int i = 0; i < numOutputs; i++)
    {
        outputLayerBias[i] = init_weights();
    }

    for (int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0, 1, 2, 3};

    int numberOfEpochs = 1000;

    // Train the neural network for a number of epochs
    for (int epoch = 0; epoch < numberOfEpochs; epoch++)
    {

        // Shuffle training set order

        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++)
        {

            // Cycle through each element in training set
            int i = trainingSetOrder[x];

            // Forward pass

            // Compute hidden layer activation
            for (int j = 0; j < numHiddenNodes; j++)
            {
                double activation = hiddenLayerBias[j];

                for (int k = 0; k < numInputs; k++)
                {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }

                // j = number of hidden nodes
                hiddenLayer[j] = sigmoid(activation);
            }

            // Compute output activation
            for (int j = 0; j < numOutputs; j++)
            {
                double activation = outputLayerBias[j];

                for (int k = 0; k < numHiddenNodes; k++)
                {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: (%g, %g)  Target: %g  Predicted: %g\n",
                   training_inputs[i][0],
                   training_inputs[i][1],
                   training_outputs[i][0],
                   outputLayer[0]);

            // Back propagation

            // Compute change in output weights

            double deltaOutput[numOutputs];

            for (int j = 0; j < numOutputs; j++)
            {
                // True label            NN Prediction
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dsigmoid(outputLayer[j]);
            }

            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++)
            {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++)
                {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dsigmoid(hiddenLayer[j]);
            }

            // Apply changes in output weights
            for (int j = 0; j < numOutputs; j++)
            {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++)
                {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            // Apply changes in hidden weights
            for (int j = 0; j < numHiddenNodes; j++)
            {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++)
                {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    // Print final weights after done training

    fputs("\n Final Hidden Weights\n", stdout);
    for (int j = 0; j < numHiddenNodes; j++)
    {
        fputs("[ ", stdout);
        for (int k = 0; k < numInputs; k++)
        {
            printf("%f ", hiddenWeights[k][j]);
        }
        fputs("] ", stdout);
    }

    fputs("]\nFinal Hidden Biases\n[", stdout);
    for (int j = 0; j < numHiddenNodes; j++)
    {
        printf("%f ", hiddenLayerBias[j]);
    }

    fputs("]Final Output Weights\n[", stdout);
    for (int j = 0; j < numOutputs; j++)
    {
        fputs("[ ", stdout);
        for (int k = 0; k < numHiddenNodes; k++)
        {
            printf("%f ", outputWeights[k][j]);
        }
        fputs("] \n", stdout);
    }

    fputs("]\nFinal Output Biases\n[", stdout);
    for (int j = 0; j < numOutputs; j++)
    {
        printf("%f ", outputLayerBias[j]);
    }

    fputs("] \n", stdout);

    return 0;
}