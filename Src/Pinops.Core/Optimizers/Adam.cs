using ILGPU;
using System.Collections.Generic;
using Pinops.Core.ComputationalKernels;
using Pinops.Core.Nodes;

namespace Pinops.Core.Optimizers
{
    public class Adam : Optimizer
    {
        internal float LearningRate { get; private set; }
        internal float Beta1 { get; private set; }
        internal float Beta2 { get; private set; }
        internal float Epsilon {get; private set;}

        internal float T { get; set; }
        internal List<float[]> MAM { get; set; }
        internal List<float[]> MAV { get; set; }

        internal List<Variable> Variables { get; private set; }

        public Adam(float learningRate = 0.001f,
                    float beta1 = 0.9f,
                    float beta2 = 0.999f,
                    float epsilon = 1e-8f)
        {
            this.LearningRate = learningRate;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;

            T = 1f;
        }

        internal override void Init(List<Variable> variables)
        {
            this.Variables = variables;
            foreach (var variable in variables)
            {
                variable.MAM = new Tensor(variable, variable.Derivatives.GetShape());
                variable.MAV = new Tensor(variable, variable.Derivatives.GetShape());
            }
        }

        internal override void Minimize(List<Variable> variables)
        {
            foreach (var variable in variables)
            {
                KernelExecutor.Execute(OptimizerKernels.AdamKernel)(new Index1D(variable.Output.Length),
                                                                    LearningRate,
                                                                    Beta1,
                                                                    Beta2,
                                                                    Epsilon,
                                                                    T,
                                                                    variable.Output.View,
                                                                    variable.Derivatives.View,
                                                                    variable.MAM.View,
                                                                    variable.MAV.View);
            }
            T += 1f;
        }
    }
}
