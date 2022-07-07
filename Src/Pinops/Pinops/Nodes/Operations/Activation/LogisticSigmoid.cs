using ILGPU;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class LogisticSigmoid : Operation
    {
        private readonly Node input;

        private readonly Tensor partialDerivativeForInput;

        internal LogisticSigmoid(Graph graph, Node input) : base(graph, input)
        {
            this.input = input;

            Output = new Tensor(this, input.Output.GetShape());
            Derivatives = new Tensor(this, Output.GetShape());
            partialDerivativeForInput = new Tensor(this, Derivatives.GetShape());
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(ActivationKernels.LogisticSigmoidKernel)(new Index1D(input.Output.Length),
                                                                            input.Output.View,
                                                                            Output.View);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(ActivationKernels.LogisticSigmoidDerivativesKernel)(new Index1D(Derivatives.Length),
                                                                                       Derivatives.View,
                                                                                       Output.View,
                                                                                       partialDerivativeForInput.View);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(input.Derivatives.Length),
                                                           input.Derivatives.View,
                                                           partialDerivativeForInput.View);
        }
    }
}
