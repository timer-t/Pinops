using ILGPU;
using ILGPU.Runtime;
using System;
using System.Linq;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class Add : Operation
    {
        private readonly Node inputA,
                              inputB;

        private readonly int threads;

        private readonly MemoryBuffer1D<int, Stride1D.Dense> inputAShapeBuffer,
                                                             inputBShapeBuffer,
                                                             resultShapeBuffer;

        internal Add(Graph graph, Node inputA, Node inputB) : base(graph, inputA, inputB)
        {
            if (!IsTensorsCompatible(inputA.GetShape(), inputB.GetShape()))
            {
                throw new Exception("Tensors are not compatible.");
            }

            this.inputA = inputA;
            this.inputB = inputB;

            var inputAShape = inputA.GetShape();
            var inputBShape = inputB.GetShape();
            var resultShape = GetResultShape(inputA.GetShape(), inputB.GetShape());

            EvenShapes(ref inputAShape, ref inputBShape);

            threads = resultShape.Aggregate(1, (a, b) => a * b);

            inputAShapeBuffer = KernelExecutor.Accelerator.Allocate1D<int>(inputAShape.Length);
            inputAShapeBuffer.CopyFromCPU(inputAShape);
            inputBShapeBuffer = KernelExecutor.Accelerator.Allocate1D<int>(inputBShape.Length);
            inputBShapeBuffer.CopyFromCPU(inputBShape);
            resultShapeBuffer = KernelExecutor.Accelerator.Allocate1D<int>(resultShape.Length);
            resultShapeBuffer.CopyFromCPU(resultShape);

            Output = new Tensor(this, resultShape);
            Derivatives = new Tensor(this, Output.GetShape());
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(BasicKernels.AddWithBroadcastingKernel)(new Index1D(threads),
                                                                           inputA.Output.View, inputAShapeBuffer.View,
                                                                           inputB.Output.View, inputBShapeBuffer.View,
                                                                           Output.View, resultShapeBuffer.View);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(inputA.Derivatives.Length),
                                                           inputA.Derivatives.View,
                                                           Derivatives.View);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(inputB.Derivatives.Length),
                                                           inputB.Derivatives.View,
                                                           Derivatives.View);
        }

        private bool IsTensorsCompatible(int[] shapeA, int[] shapeB)
        {
            var iA = shapeA.Length - 1;
            var iB = shapeB.Length - 1;
            while (iA >= 0 && iB >= 0)
            {
                if (shapeA[iA] != shapeB[iB] &&
                    shapeA[iA] != 1 &&
                    shapeB[iB] != 1)
                {
                    return false;
                }

                iA--;
                iB--;
            }

            return true;
        }

        private int[] GetResultShape(int[] shapeA, int[] shapeB)
        {
            var longestLength = shapeA.Length > shapeB.Length ? shapeA.Length : shapeB.Length;

            var resultShape = new int[longestLength];

            var i = resultShape.Length - 1;
            var iA = shapeA.Length - 1;
            var iB = shapeB.Length - 1;
            while (i >= 0)
            {
                if (iA >= 0 && iB >= 0)
                {
                    resultShape[i] = shapeA[iA] > 1 ? shapeA[iA] : shapeB[iB];
                }
                else if (iA < 0)
                {
                    resultShape[i] = shapeB[iB];
                }
                else if (iB < 0)
                {
                    resultShape[i] = shapeA[iA];
                }

                i--;
                iA--;
                iB--;
            }

            return resultShape;
        }

        private void EvenShapes(ref int[] shapeA, ref int[] shapeB)
        {
            if (shapeA.Length < shapeB.Length)
            {
                var newInputAShape = new int[shapeB.Length];
                var difference = shapeB.Length - shapeA.Length;
                for (int i = 0; i < difference; i++)
                {
                    newInputAShape[i] = 1;
                }
                for (int i = difference; i < newInputAShape.Length; i++)
                {
                    newInputAShape[i] = shapeA[i - difference];
                }

                shapeA = newInputAShape;
            }
            else if(shapeB.Length < shapeA.Length)
            {
                EvenShapes(ref shapeB, ref shapeA);
            }
        }
    }
}
