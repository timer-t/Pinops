using ILGPU;
using System;
using System.Linq;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class MSE : Operation, ILossOperation
    {
        private readonly Node predicted,
                              observed;

        private readonly Tensor partialDerivativeForPredicted;

        private readonly int batchSize,
                             valuesCount;

        internal MSE(Graph graph, Node predicted, Placeholder observed) : base(graph, predicted, observed)
        {
            if (!Enumerable.SequenceEqual(predicted.Output.GetShape(), observed.Output.GetShape()))
            {
                throw new Exception("Dimensions of input tensors must be equal.");
            }
            if (predicted.Output.Length != observed.Output.Length)
            {
                throw new Exception("Lengths of input tensors must be equal.");
            }

            this.predicted = predicted;
            this.observed = observed;

            batchSize = predicted.Output.GetShape(0);
            valuesCount = predicted.Output.GetShape(1);

            Output = new Tensor(this, batchSize);

            partialDerivativeForPredicted = new Tensor(this, predicted.Output.GetShape());
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(LossKernels.MSEKernel)(new Index1D(batchSize),
                                                          batchSize,
                                                          valuesCount,
                                                          predicted.Output.View,
                                                          observed.Output.View,
                                                          Output.View);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(LossKernels.MSEDerivativesKernel)(new Index2D(valuesCount, batchSize),
                                                                     batchSize,
                                                                     valuesCount,
                                                                     predicted.Output.View,
                                                                     observed.Output.View,
                                                                     partialDerivativeForPredicted.View);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(predicted.Derivatives.Length),
                                                           predicted.Derivatives.View,
                                                           partialDerivativeForPredicted.View);
        }
    }
}
