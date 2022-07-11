using ILGPU;
using System;
using System.Linq;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class Sum : Operation
    {
        private readonly Node inputNodeA,
                              inputNodeB;

        internal Sum(Graph graph, Node inputNodeA, Node inputNodeB) : base(graph, inputNodeA, inputNodeB)
        {
            if (!Enumerable.SequenceEqual(inputNodeA.Output.GetShape(), inputNodeB.Output.GetShape()))
            {
                throw new Exception("Dimensions of input tensors must be equal.");
            }
            if (inputNodeA.Output.Length != inputNodeB.Output.Length)
            {
                throw new Exception("Lengths of input tensors must be equal.");
            }

            this.inputNodeA = inputNodeA;
            this.inputNodeB = inputNodeB;

            Output = new Tensor(this, inputNodeA.Output.GetShape());
            Derivatives = new Tensor(this, Output.GetShape());
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(BasicKernels.SumKernel)(new Index1D(inputNodeA.Output.Length),
                                                           inputNodeA.Output.View,
                                                           inputNodeB.Output.View,
                                                           Output.View);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(inputNodeA.Derivatives.Length),
                                                           inputNodeA.Derivatives.View,
                                                           Derivatives.View);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(inputNodeB.Derivatives.Length),
                                                           inputNodeB.Derivatives.View,
                                                           Derivatives.View);
        }
    }
}
