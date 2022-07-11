using ILGPU;
using ILGPU.Algorithms;

namespace Pinops.Core.ComputationalKernels
{
    internal static class ActivationKernels
    {
        internal static void LogisticSigmoidKernel(Index1D index,
                                                   ArrayView<float> input,
                                                   ArrayView<float> result)
        {
            result[index] = 1.0f / (1.0f + XMath.Pow(XMath.E, -input[index]));
        }

        internal static void LogisticSigmoidDerivativesKernel(Index1D index,
                                                              ArrayView<float> derivatives,
                                                              ArrayView<float> output,
                                                              ArrayView<float> result)
        {
            result[index] = derivatives[index] * output[index] * (1.0f - output[index]);
        }

        internal static void LeakyReLUKernel(Index1D index,
                                             ArrayView<float> input,
                                             ArrayView<float> result,
                                             float negativeSlope)
        {
            result[index] = input[index] > 0 ? input[index] : negativeSlope * input[index];
        }

        internal static void LeakyReLUDerivativesKernel(Index1D index,
                                                        ArrayView<float> derivatives,
                                                        ArrayView<float> input,
                                                        ArrayView<float> result,
                                                        float negativeSlope)
        {
            var val = input[index] > 0 ? 1f : negativeSlope;
            result[index] = derivatives[index] * val;
        }
    }
}
