using ILGPU;
using System.Linq;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class OneHot : Operation
    {
        private readonly Node indices;
        private readonly float[] indicesArray;
        private readonly float indicesMaxValue;

        internal OneHot(Graph graph, Node indices) : base(graph, indices)
        {
            this.indices = indices;

            indicesArray = indices.Output.As1DArray();
            indicesMaxValue = indicesArray.Max() + 1;

            int[] outputShape = new int[indices.Output.GetShape().Length + 1];

            for (int i = 0; i < indices.Output.GetShape().Length; i++)
            {
                outputShape[i] = indices.Output.GetShape(i);
            }
            outputShape[^1] = (int)indicesMaxValue;

            Output = new Tensor(this, outputShape);
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(BasicKernels.OneHotKernel)(new Index1D(indicesArray.Length),
                                                              indices.Output.View,
                                                              Output.View,
                                                              indicesMaxValue);
        }

        internal override void Backward()
        {
            throw new System.NotImplementedException();
        }
    }
}
