using ILGPU;
using ILGPU.Algorithms;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.ComputationalKernels
{
    internal static class BatchNorm2DKernels
    {
        internal static void MeanForwardKernel(Index1D index,
                                               ArrayView<float> data,
                                               ArrayView<float> mean,
                                               ArrayView<float> storedMean,
                                               float momentum,
                                               int batchSize,
                                               int channelsCount,
                                               int channelLength)
        {
            int batchInstanceLength = channelsCount * channelLength;
            for (int b = 0; b < batchSize; b++)
            {
                for (int cl = 0; cl < channelLength; cl++)
                {
                    int i = (batchInstanceLength * b) + (channelLength * index + cl);
                    mean[index] += data[i];
                }
            }
            mean[index] = mean[index] * (1f / (channelLength * batchSize));

            if (storedMean[index] == 0f)
            {
                storedMean[index] = mean[index];
            }
            else
            {
                storedMean[index] = storedMean[index] * momentum + mean[index] * (1f - momentum);
            }
        }

        internal static void VarianceForwardKernel(Index1D index,
                                                   ArrayView<float> data,
                                                   ArrayView<float> mean,
                                                   ArrayView<float> variance,
                                                   ArrayView<float> storedVariance,
                                                   float momentum,
                                                   int batchSize,
                                                   int channelsCount,
                                                   int channelLength)
        {
            int batchInstanceLength = channelsCount * channelLength;
            for (int b = 0; b < batchSize; b++)
            {
                for (int cl = 0; cl < channelLength; cl++)
                {
                    int i = (batchInstanceLength * b) + (channelLength * index + cl);
                    variance[index] += XMath.Pow(data[i] - mean[index], 2f);
                }
            }
            variance[index] = variance[index] * (1f / (channelLength * batchSize));

            if (storedVariance[index] == 0f)
            {
                storedVariance[index] = variance[index];
            }
            else
            {
                storedVariance[index] = storedVariance[index] * momentum + variance[index] * (1f - momentum);
            }
        }

        internal static void NormalizeForwardKernel(Index1D index,
                                                    ArrayView<float> data,
                                                    ArrayView<float> mean,
                                                    ArrayView<float> variance,
                                                    ArrayView<float> normalized,
                                                    float epsilon,
                                                    int channelsCount,
                                                    int channelLength)
        {
            int channelIndex = (index / channelLength) % channelsCount;
            normalized[index] = (data[index] - mean[channelIndex]) / XMath.Sqrt(variance[channelIndex] + epsilon);
        }

        internal static void OutputForwardKernel(Index1D index,
                                                 ArrayView<float> normalized,
                                                 ArrayView<float> gamma,
                                                 ArrayView<float> beta,
                                                 ArrayView<float> output,
                                                 int channelsCount,
                                                 int channelLength)
        {
            int channelIndex = (index / channelLength) % channelsCount;
            output[index] = gamma[channelIndex] * normalized[index] + beta[channelIndex];
        }

        internal static void OutputBackwardKernel(Index1D index,
                                                  ArrayView<float> derivatives,
                                                  ArrayView<float> normalized,
                                                  ArrayView<float> gamma,
                                                  ArrayView<float> partialDerivativeForGamma,
                                                  ArrayView<float> partialDerivativeForBeta,
                                                  ArrayView<float> outputDelta,
                                                  int channelsCount,
                                                  int channelLength)
        {
            int channelIndex = (index / channelLength) % channelsCount;
            partialDerivativeForGamma[channelIndex] += derivatives[index] * normalized[index];
            partialDerivativeForBeta[channelIndex] += derivatives[index];
            outputDelta[index] = derivatives[index] * gamma[channelIndex];
        }

        internal static void MeanBackwardKernel(Index1D index,
                                                ArrayView<float> outputDelta,
                                                ArrayView<float> variance,
                                                ArrayView<float> meanDelta,
                                                float epsilon,
                                                int batchSize,
                                                int channelsCount,
                                                int channelLength)
        {
            int batchInstanceLength = channelsCount * channelLength;
            for (int b = 0; b < batchSize; b++)
            {
                for (int cl = 0; cl < channelLength; cl++)
                {
                    int i = (batchInstanceLength * b) + (channelLength * index + cl);
                    meanDelta[index] += outputDelta[i];
                }
            }
            meanDelta[index] = meanDelta[index] * (-1f / XMath.Sqrt(variance[index] + epsilon));
        }

        internal static void VarianceBackwardKernel(Index1D index,
                                                    ArrayView<float> outputDelta,
                                                    ArrayView<float> data,
                                                    ArrayView<float> mean,
                                                    ArrayView<float> variance,
                                                    ArrayView<float> varianceDelta,
                                                    float epsilon,
                                                    int batchSize,
                                                    int channelsCount,
                                                    int channelLength)
        {
            int batchInstanceLength = channelsCount * channelLength;
            for (int b = 0; b < batchSize; b++)
            {
                for (int cl = 0; cl < channelLength; cl++)
                {
                    int i = (batchInstanceLength * b) + (channelLength * index + cl);
                    varianceDelta[index] += outputDelta[i] * (data[i] - mean[index]);
                }
            }
            varianceDelta[index] = varianceDelta[index] * -0.5f * XMath.Pow(variance[index] + epsilon, -1.5f);
        }

        internal static void NormalizeBackwardKernel(Index1D index,
                                                     ArrayView<float> outputDelta,
                                                     ArrayView<float> meanDelta,
                                                     ArrayView<float> varianceDelta,
                                                     ArrayView<float> data,
                                                     ArrayView<float> mean,
                                                     ArrayView<float> variance,
                                                     ArrayView<float> normalizedDelta,
                                                     float epsilon,
                                                     int batchSize,
                                                     int channelsCount,
                                                     int channelLength)
        {
            int channelIndex = (index / channelLength) % channelsCount;
            normalizedDelta[index] = outputDelta[index] * (1f / XMath.Sqrt(variance[channelIndex] + epsilon)) +
                                     meanDelta[channelIndex] * (1f / (channelLength * batchSize)) +
                                     varianceDelta[channelIndex] * (2f * (data[index] - mean[channelIndex]) / (channelLength * batchSize));
        }
    }
}
