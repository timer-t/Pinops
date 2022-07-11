using ILGPU;
using ILGPU.Algorithms;

namespace Pinops.Core.ComputationalKernels
{
    internal static class LossKernels
    {
        internal static void MSEKernel(Index1D index,
                                       int batchSize,
                                       int valuesCount,
                                       ArrayView<float> predicted,
                                       ArrayView<float> observed,
                                       ArrayView<float> result)
        {
            var sum = 0.0f;

            for (var i = 0; i < valuesCount; i++)
                sum += XMath.Pow((observed[valuesCount * index + i] - predicted[valuesCount * index + i]), 2.0f);

            result[index] = 0.5f * sum;
        }

        internal static void MSEDerivativesKernel(Index2D index,
                                                  int batchSize,
                                                  int valuesCount,
                                                  ArrayView<float> predicted,
                                                  ArrayView<float> observed,
                                                  ArrayView<float> result)
        {
            var x = index.X;
            var y = index.Y;

            result[valuesCount * y + x] = predicted[valuesCount * y + x] - observed[valuesCount * y + x];
        }
    }
}
