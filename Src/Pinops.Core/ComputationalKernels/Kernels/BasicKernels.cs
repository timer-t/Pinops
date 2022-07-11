using ILGPU;

namespace Pinops.Core.ComputationalKernels
{
    internal static class BasicKernels
    {
        internal static void AddKernel(Index1D index, ArrayView<float> destination, ArrayView<float> source)
        {
            destination[index] += source[index];
        }

        internal static void SumKernel(Index1D index, ArrayView<float> aView, ArrayView<float> bView, ArrayView<float> result)
        {
            result[index] = aView[index] + bView[index];
        }

        internal static void AddWithBroadcastingKernel(Index1D index,
                                                       ArrayView<float> inputA, ArrayView<int> inputAShape,
                                                       ArrayView<float> inputB, ArrayView<int> inputBShape,
                                                       ArrayView<float> result, ArrayView<int> resultShape)
        {
            int indexRemains = index;

            int indexA = 0;
            int indexB = 0;

            for (int dimI = 0; dimI < resultShape.Length; dimI++)
            {
                int sum = 1;
                int sumA = 1;
                int sumB = 1;

                for (int sumI = dimI + 1; sumI < resultShape.Length; sumI++)
                {
                    sum *= resultShape[sumI];
                    sumA *= inputAShape[sumI];
                    sumB *= inputBShape[sumI];
                }

                int coor;

                if (dimI == resultShape.Length - 1)
                {
                    coor = indexRemains;
                }
                else
                {
                    coor = indexRemains / sum;
                }

                indexRemains -= coor * sum;

                int coorA = inputAShape[dimI] == 1 ? 0 : coor;
                int coorB = inputBShape[dimI] == 1 ? 0 : coor;

                indexA += coorA * sumA;
                indexB += coorB * sumB;
            }

            result[index] = inputA[indexA] + inputB[indexB];
        }

        internal static void OneHotKernel(Index1D index,
                                          ArrayView<float> indices,
                                          ArrayView<float> output,
                                          float indicesMaxValue)
        {
            int x = (int)(index * indicesMaxValue + indices[index]);
            output[x] = 1;
        }
    }
}
