using ILGPU;
using System.Collections.Generic;
using Pinops.Core.ComputationalKernels;
using Pinops.Core.Nodes;

namespace Pinops.Core.Optimizers
{
    public class SGD : Optimizer
    {
        internal readonly float LearningRate;

        public SGD(float learningRate)
        {
            this.LearningRate = learningRate;
        }

        internal override void Init(List<Variable> variables)
        {

        }

        internal override void Minimize(List<Variable> variables)
        {
            foreach (var variable in variables)
            {
                KernelExecutor.Execute(OptimizerKernels.SGDKernel)(new Index1D(variable.Output.Length),
                                                                   LearningRate,
                                                                   variable.Output.View,
                                                                   variable.Derivatives.View);
            }
        }
    }
}
