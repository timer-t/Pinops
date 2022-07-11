using ILGPU;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class LeakyReLU : Operation
    {
        private readonly Node input;

        internal readonly float NegativeSlope;

        private readonly Tensor partialDerivativeForInput;

        internal LeakyReLU(Graph graph, Node input, float negativeSlope) : base(graph, input)
        {
            this.input = input;
            this.NegativeSlope = negativeSlope;

            Output = new Tensor(this, input.Output.GetShape());
            Derivatives = new Tensor(this, Output.GetShape());
            partialDerivativeForInput = new Tensor(this, Derivatives.GetShape());
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(ActivationKernels.LeakyReLUKernel)(new Index1D(input.Output.Length),
                                                                      input.Output.View,
                                                                      Output.View,
                                                                      NegativeSlope);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(ActivationKernels.LeakyReLUDerivativesKernel)(new Index1D(Derivatives.Length),
                                                                                 Derivatives.View,
                                                                                 input.Output.View,
                                                                                 partialDerivativeForInput.View,
                                                                                 NegativeSlope);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(input.Derivatives.Length),
                                                           input.Derivatives.View,
                                                           partialDerivativeForInput.View);
        }
    }
}
