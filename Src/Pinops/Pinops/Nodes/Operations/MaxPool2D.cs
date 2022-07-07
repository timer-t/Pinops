using ILGPU;
using ILGPU.Runtime;
using System;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class MaxPool2D : Operation
    {
        private readonly Node input;

        internal readonly int Kernel,
                              Stride,
                              Padding,
                              Dilation;

        private readonly Tensor partialDerivativeForInput;

        private readonly int forward_num_kernels,
                             backward_num_kernels;

        private readonly MemoryBuffer1D<int, Stride1D.Dense> parametersBuffer;
        private readonly MemoryBuffer1D<float, Stride1D.Dense> topMaskBuffer;

        internal MaxPool2D(Graph graph, Node input,
                           int kernel, int stride = 1, int padding = 0) : base(graph, input)
        {
            if (input.Output.GetShape().Length != 4)
            {
                throw new Exception("Input should be of (N,C,H,W) shape.");
            }

            this.input = input;

            this.Kernel = kernel;
            this.Stride = stride;
            this.Padding = padding;

            int pooledHeight = (input.Output.GetShape(2) + 2 * padding - (kernel - 1) - 1) / stride + 1;
            int pooledWidth = (input.Output.GetShape(3) + 2 * padding - (kernel - 1) - 1) / stride + 1;

            Output = new Tensor(this, input.Output.GetShape(0), input.Output.GetShape(1), pooledHeight, pooledWidth);
            Derivatives = new Tensor(this, Output.GetShape());
            if (!(input is Placeholder))
            {
                partialDerivativeForInput = new Tensor(this, input.Derivatives.GetShape());
            }

            var batchSize = input.Output.GetShape(0);
            var inputChannels = input.Output.GetShape(1);

            int inputHeight = input.Output.GetShape(2),
                kernel_h = kernel,
                pad_h = padding,
                stride_h = stride;

            int inputWidth = input.Output.GetShape(3),
                kernel_w = kernel,
                pad_w = padding,
                stride_w = stride;

            forward_num_kernels = batchSize * inputChannels * pooledHeight * pooledWidth;
            backward_num_kernels = batchSize * inputChannels * inputHeight * inputWidth;

            int[] parameters = new int[11]
            {
                inputChannels,
                inputHeight, inputWidth,
                pooledHeight, pooledWidth,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w
            };

            parametersBuffer = KernelExecutor.Accelerator.Allocate1D<int>(parameters.Length);
            parametersBuffer.CopyFromCPU(parameters);

            topMaskBuffer = KernelExecutor.Accelerator.Allocate1D<float>(Output.Length);
            topMaskBuffer.MemSetToZero();
        }

        internal override void Forward()
        {
            KernelExecutor.Execute(PoolingKernels.MaxPoolForwardKernel)(new Index1D(forward_num_kernels),
                                                                        input.Output.View,
                                                                        parametersBuffer.View,
                                                                        Output.View,
                                                                        topMaskBuffer.View);
        }

        internal override void Backward()
        {
            if (!(input is Placeholder))
            {
                KernelExecutor.Execute(PoolingKernels.MaxPoolBackwardKernel)(new Index1D(backward_num_kernels),
                                                                             Derivatives.View,
                                                                             parametersBuffer.View,
                                                                             partialDerivativeForInput.View,
                                                                             topMaskBuffer.View);

                KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(input.Derivatives.Length),
                                                               input.Derivatives.View,
                                                               partialDerivativeForInput.View);
            }
        }
    }
}
