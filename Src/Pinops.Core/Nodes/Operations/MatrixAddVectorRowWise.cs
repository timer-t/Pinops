using ILGPU;
using System;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    [Obsolete("Use \"Add\" operation instead.", true)]
    internal class MatrixAddVectorRowWise : Operation
    {
        private readonly Node matrix,
                              vector;

        private readonly int matrixHeight,
                             matrixWidth;

        internal MatrixAddVectorRowWise(Graph graph, Node matrix, Node vector, int matrixHeight, int matrixWidth) : base(graph, matrix, vector)
        {
            this.matrix = matrix;
            this.vector = vector;
            this.matrixHeight = matrixHeight;
            this.matrixWidth = matrixWidth;

            Output = new Tensor(this, matrix.Output.GetShape());
            Derivatives = new Tensor(this, Output.GetShape());
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(MatrixKernels.MatrixAddVectorRowWiseKernel)(new Index2D(matrixWidth,
                                                                                           matrixHeight),
                                                                               matrixWidth,
                                                                               matrix.Output.View,
                                                                               vector.Output.View,
                                                                               Output.View);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(matrix.Derivatives.Length),
                                                           matrix.Derivatives.View,
                                                           Derivatives.View);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(vector.Derivatives.Length),
                                                           vector.Derivatives.View,
                                                           Derivatives.View);
        }
    }
}
