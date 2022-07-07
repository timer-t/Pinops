using ILGPU;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class MatMul : Operation
    {
        private readonly Node inputA,
                              inputB;

        private readonly Tensor partialDerivativeForInputA,
                                partialDerivativeForInputB;

        internal MatMul(Graph graph, Node inputA, Node inputB) : base(graph, inputA, inputB)
        {
            this.inputA = inputA;
            this.inputB = inputB;

            Output = new Tensor(this, inputA.Output.GetShape(0), inputB.Output.GetShape(1));
            Derivatives = new Tensor(this, Output.GetShape());
            if (!(inputA is Placeholder))
            {
                partialDerivativeForInputA = new Tensor(this, inputA.Derivatives.GetShape());
            }
            if (!(inputB is Placeholder))
            {
                partialDerivativeForInputB = new Tensor(this, inputB.Derivatives.GetShape());
            }
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(MatrixKernels.MatrixMultiplyKernel)(new Index2D(Output.GetShape(1),
                                                                                   Output.GetShape(0)),
                                                                       inputA.Output.View,
                                                                       inputA.Output.GetShape(0),
                                                                       inputA.Output.GetShape(1),
                                                                       inputB.Output.View,
                                                                       inputB.Output.GetShape(0),
                                                                       inputB.Output.GetShape(1),
                                                                       Output.View,
                                                                       Output.GetShape(0),
                                                                       Output.GetShape(1),
                                                                       0,
                                                                       0,
                                                                       0);
        }

        internal override void Backward()
        {
            if (!(inputA is Placeholder))
            {
                KernelExecutor.Execute(MatrixKernels.MatrixMultiplyRightTransposedKernel)(new Index2D(partialDerivativeForInputA.GetShape(1),
                                                                                                      partialDerivativeForInputA.GetShape(0)),
                                                                                          Derivatives.View,
                                                                                          Derivatives.GetShape(0),
                                                                                          Derivatives.GetShape(1),
                                                                                          inputB.Output.View,
                                                                                          inputB.Output.GetShape(0),
                                                                                          inputB.Output.GetShape(1),
                                                                                          partialDerivativeForInputA.View,
                                                                                          partialDerivativeForInputA.GetShape(0),
                                                                                          partialDerivativeForInputA.GetShape(1),
                                                                                          0,
                                                                                          0,
                                                                                          0);

                KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(inputA.Derivatives.Length),
                                                               inputA.Derivatives.View,
                                                               partialDerivativeForInputA.View);
            }
            if (!(inputB is Placeholder))
            {
                KernelExecutor.Execute(MatrixKernels.MatrixMultiplyLeftTransposedKernel)(new Index2D(partialDerivativeForInputB.GetShape(1),
                                                                                                     partialDerivativeForInputB.GetShape(0)),
                                                                                         inputA.Output.View,
                                                                                         inputA.Output.GetShape(0),
                                                                                         inputA.Output.GetShape(1),
                                                                                         Derivatives.View,
                                                                                         Derivatives.GetShape(0),
                                                                                         Derivatives.GetShape(1),
                                                                                         partialDerivativeForInputB.View,
                                                                                         partialDerivativeForInputB.GetShape(0),
                                                                                         partialDerivativeForInputB.GetShape(1),
                                                                                         0,
                                                                                         0,
                                                                                         0);

                KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(inputB.Derivatives.Length),
                                                               inputB.Derivatives.View,
                                                               partialDerivativeForInputB.View);
            }
        }
    }
}
