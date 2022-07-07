using ILGPU;
using ILGPU.Algorithms;

namespace Pinops.Core.ComputationalKernels
{
    internal static class OptimizerKernels
    {
        internal static void SGDKernel(Index1D index,
                                       float learningRate,
                                       ArrayView<float> variables,
                                       ArrayView<float> derivatives)
        {
            variables[index] -= learningRate * derivatives[index];
        }

        internal static void AdamKernel(Index1D index,
                                        float learningRate,
                                        float beta1,
                                        float beta2,
                                        float epsilon,
                                        float t,
                                        ArrayView<float> variables,
                                        ArrayView<float> derivatives,
                                        ArrayView<float> mam,
                                        ArrayView<float> mav)
        {
            // Calculation of moving averages
            mam[index] = beta1 * mam[index] + (1 - beta1) * derivatives[index];
            mav[index] = beta2 * mav[index] + (1 - beta2) * XMath.Pow(derivatives[index], 2);

            // Correction of the moving averages
            float mCorr = mam[index] / (1 - XMath.Pow(beta1, t));
            float vCorr = mav[index] / (1 - XMath.Pow(beta2, t));

            // Update
            variables[index] = variables[index] - learningRate * (mCorr / (XMath.Sqrt(vCorr) + epsilon));
        }
    }
}
