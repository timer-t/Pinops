using ILGPU;
using System;

namespace Pinops.Core.ComputationalKernels
{
    internal static class MatrixKernels
    {
        internal static void MatrixMultiplyKernel(Index2D index,
                                                  ArrayView<float> aView, int aHeight, int aWidth,
                                                  ArrayView<float> bView, int bHeight, int bWidth,
                                                  ArrayView<float> cView, int cHeight, int cWidth,
                                                  int aViewOffset = 0,
                                                  int bViewOffset = 0,
                                                  int cViewOffset = 0)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < bHeight; i++)
                sum += aView[(y * bHeight + i) + aViewOffset] * bView[(i * bWidth + x) + bViewOffset];

            cView[(y * cWidth + x) + cViewOffset] = sum;
        }

        internal static void MatrixMultiplyLeftTransposedKernel(Index2D index,
                                                                ArrayView<float> aView, int aHeight, int aWidth,
                                                                ArrayView<float> bView, int bHeight, int bWidth,
                                                                ArrayView<float> cView, int cHeight, int cWidth,
                                                                int aViewOffset = 0,
                                                                int bViewOffset = 0,
                                                                int cViewOffset = 0)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < bHeight; i++)
                sum += aView[(i * aWidth + y) + aViewOffset] * bView[(i * bWidth + x) + bViewOffset];

            cView[(y * cWidth + x) + cViewOffset] = sum;
        }

        internal static void MatrixMultiplyRightTransposedKernel(Index2D index,
                                                                 ArrayView<float> aView, int aHeight, int aWidth,
                                                                 ArrayView<float> bView, int bHeight, int bWidth,
                                                                 ArrayView<float> cView, int cHeight, int cWidth,
                                                                 int aViewOffset = 0,
                                                                 int bViewOffset = 0,
                                                                 int cViewOffset = 0)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < aWidth; i++)
                sum += aView[(y * aWidth + i) + aViewOffset] * bView[(x * bWidth + i) + bViewOffset];

            cView[(y * cWidth + x) + cViewOffset] = sum;
        }

        [Obsolete("Use \"Add\" operation instead.", true)]
        internal static void MatrixAddVectorRowWiseKernel(Index2D index,
                                                          int matrixWidth,
                                                          ArrayView<float> matrix,
                                                          ArrayView<float> vector,
                                                          ArrayView<float> result)
        {
            var x = index.X;
            var y = index.Y;

            result[matrixWidth * y + x] = matrix[matrixWidth * y + x] + vector[x];
        }
    }
}
