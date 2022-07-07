using ILGPU;
using ILGPU.Runtime;
using System;
using System.Linq;
using Pinops.Core.ComputationalKernels;
using Pinops.Core.Interfaces;
using Pinops.Core.Nodes;

namespace Pinops.Core
{
    public class Tensor : ITensorNodeProvider
    {
        private readonly MemoryBuffer1D<float, Stride1D.Dense> buffer;
        private int[] shape;

        internal Tensor(Node parentNode, params int[] shape)
        {
            ParentNode = parentNode;
            this.shape = shape;
            buffer = KernelExecutor.Accelerator.Allocate1D<float>(Length);
            Clear();
        }
        internal Tensor(Node parentNode, float[,] matrix) : this(parentNode, matrix.GetLength(0), matrix.GetLength(1))
        {
            Load(matrix);
        }

        public Tensor GetTensor()
        {
            return this;
        }
        public Node GetNode()
        {
            return ParentNode;
        }

        internal int Length
        {
            get
            {
                return shape.Aggregate((a, b) => a * b);
            }
        }
        internal ArrayView<float> View
        {
            get
            {
                return buffer.View;
            }
        }
        internal Node ParentNode { get; private set; }

        public Tensor this[int i]
        {
            get
            {
                int[] newShape = null;
                int[] thisShape = this.GetShape();
                int sourceOffset = 0;

                if (i < 0 || i >= thisShape[0])
                {
                    throw new IndexOutOfRangeException();
                }
                if (thisShape.Length >= 2)
                {
                    newShape = thisShape.Skip(1).ToArray();
                    sourceOffset = thisShape.Skip(1).Aggregate((a, b) => a * b);
                    sourceOffset *= i;
                }
                if (thisShape.Length == 1)
                {
                    newShape = new int[] { 1 };
                    sourceOffset = i;
                }

                var newTensor = new Tensor(this.ParentNode, newShape);
                newTensor.Load(this.buffer, sourceOffset);

                return newTensor;
            }
        }

        internal void Load(float[,] matrix)
        {
            Load(Helper.Flatten(matrix));
        }
        internal void Load(float[] data)
        {
            buffer.CopyFromCPU(data);
        }
        internal void Load(MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer, int sourceOffset)
        {
            sourceBuffer.View.SubView(sourceOffset, buffer.Extent).CopyTo(buffer);
        }
        internal void Clear()
        {
            buffer.MemSetToZero();
        }
        public float[,] As2DArray()
        {
            if (shape.Length != 2)
            {
                throw new Exception("Shape is not two-dimensional.");
            }

            return Helper.ReFlatten(buffer.GetAsArray1D(), shape[0], shape[1]);
        }
        public float[] As1DArray()
        {
            return buffer.GetAsArray1D();
        }
        public int[] GetShape()
        {
            return shape;
        }
        public int GetShape(int index)
        {
            return shape[index];
        }
        public void SetShape(int[] newShape)
        {
            shape = newShape;
        }
    }
}
