using ILGPU;
using ILGPU.Runtime;
using System;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class Conv2D : Operation
    {
        private readonly Node data,
                              weights;

        internal readonly int Channels,
                              Kernel,
                              Stride,
                              Padding,
                              Dilation;

        private readonly Tensor partialDerivativeForData,
                                partialDerivativeForWeights;

        private readonly int batchSize,
                             inputChannels,
                             im2col_num_kernels,
                             col2im_num_kernels,
                             inputImageLength,
                             outputImageLength,
                             im2colHeight,
                             im2colWidth,
                             im2colLength;

        private readonly MemoryBuffer1D<int, Stride1D.Dense> parametersBuffer;
        private readonly MemoryBuffer1D<float, Stride1D.Dense> im2colResultBuffer,
                                                               col2imInputBuffer;

        internal Conv2D(Graph graph,
                        Node data, Variable weights,
                        int channels, int kernel, int stride = 1, int padding = 0, int dilation = 1) : base(graph, data, weights)
        {
            if (data.Output.GetShape().Length != 4)
            {
                throw new Exception("Input should be of (N,C,H,W) shape.");
            }

            this.data = data;
            this.weights = weights;

            this.Channels = channels;
            this.Kernel = kernel;
            this.Stride = stride;
            this.Padding = padding;
            this.Dilation = dilation;

            int outputHeight = (data.Output.GetShape(2) + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
            int outputWidth = (data.Output.GetShape(3) + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

            Output = new Tensor(this, data.Output.GetShape(0), channels, outputHeight, outputWidth);
            Derivatives = new Tensor(this, Output.GetShape());
            if (!(data is Placeholder))
            {
                partialDerivativeForData = new Tensor(this, data.Derivatives.GetShape());
            }
            partialDerivativeForWeights = new Tensor(this, weights.Derivatives.GetShape());

            batchSize = data.Output.GetShape(0);
            inputChannels = data.Output.GetShape(1);

            int inputHeight = data.Output.GetShape(2),
                kernel_h = kernel,
                pad_h = padding,
                stride_h = stride,
                dilation_h = dilation;

            int inputWidth = data.Output.GetShape(3),
                kernel_w = kernel,
                pad_w = padding,
                stride_w = stride,
                dilation_w = dilation;

            int height_col = (inputHeight + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
            int width_col = (inputWidth + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

            im2col_num_kernels = inputChannels * height_col * width_col;
            col2im_num_kernels = inputChannels * inputHeight * inputWidth;

            int[] parameters = new int[12]
            {
                inputHeight, inputWidth,
                kernel_h, kernel_w,
                pad_h, pad_w,
                stride_h, stride_w,
                dilation_h, dilation_w,
                height_col, width_col
            };

            parametersBuffer = KernelExecutor.Accelerator.Allocate1D<int>(parameters.Length);
            parametersBuffer.CopyFromCPU(parameters);

            im2colHeight = kernel * kernel * inputChannels;
            im2colWidth = height_col * width_col;
            im2colLength = im2colHeight * im2colWidth;

            im2colResultBuffer = KernelExecutor.Accelerator.Allocate1D<float>(batchSize * im2colHeight * im2colWidth);
            col2imInputBuffer = KernelExecutor.Accelerator.Allocate1D<float>(im2colHeight * im2colWidth);

            inputImageLength = data.Output.GetShape(1) * inputHeight * inputWidth;
            outputImageLength = Output.GetShape(1) * Output.GetShape(2) * Output.GetShape(3);
        }

        internal override void Forward()
        {
            int data_im_offset = 0;
            int data_col_offset = 0;
            int matmul_result_offset = 0;
            for (int i = 0; i < batchSize; i++)
            {
                KernelExecutor.Execute(ConvolutionKernels.Im2ColKernel)(new Index1D(im2col_num_kernels),
                                                                        data.Output.View,
                                                                        parametersBuffer.View,
                                                                        im2colResultBuffer.View,
                                                                        data_im_offset,
                                                                        data_col_offset);
                KernelExecutor.Execute(MatrixKernels.MatrixMultiplyKernel)(new Index2D(Output.GetShape(2) * Output.GetShape(3),
                                                                                       Output.GetShape(1)),
                                                                           weights.Output.View,
                                                                           weights.Output.GetShape(0),
                                                                           weights.Output.GetShape(1),
                                                                           im2colResultBuffer.View,
                                                                           im2colHeight,
                                                                           im2colWidth,
                                                                           Output.View,
                                                                           Output.GetShape(1),
                                                                           Output.GetShape(2) * Output.GetShape(3),
                                                                           0,
                                                                           data_col_offset,
                                                                           matmul_result_offset);
                data_im_offset += inputImageLength;
                data_col_offset += im2colLength;
                matmul_result_offset += outputImageLength;
            }
        }

        internal override void Backward()
        {
            int data_im_offset = 0;
            int data_col_offset = 0;
            int output_offset = 0;
            for (int i = 0; i < batchSize; i++)
            {
                KernelExecutor.Execute(MatrixKernels.MatrixMultiplyRightTransposedKernel)(new Index2D(partialDerivativeForWeights.GetShape(1),
                                                                                                      partialDerivativeForWeights.GetShape(0)),
                                                                                          Derivatives.View,
                                                                                          Derivatives.GetShape(1),
                                                                                          Derivatives.GetShape(2) * Derivatives.GetShape(3),
                                                                                          im2colResultBuffer.View,
                                                                                          im2colHeight,
                                                                                          im2colWidth,
                                                                                          partialDerivativeForWeights.View,
                                                                                          partialDerivativeForWeights.GetShape(0),
                                                                                          partialDerivativeForWeights.GetShape(1),
                                                                                          output_offset,
                                                                                          data_col_offset,
                                                                                          0);

                KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(weights.Derivatives.Length),
                                                               weights.Derivatives.View,
                                                               partialDerivativeForWeights.View);

                if (!(data is Placeholder))
                {
                    KernelExecutor.Execute(MatrixKernels.MatrixMultiplyLeftTransposedKernel)(new Index2D(im2colWidth,
                                                                                                        im2colHeight),
                                                                                             weights.Output.View,
                                                                                             weights.Output.GetShape(0),
                                                                                             weights.Output.GetShape(1),
                                                                                             Derivatives.View,
                                                                                             Derivatives.GetShape(1),
                                                                                             Derivatives.GetShape(2) * Derivatives.GetShape(3),
                                                                                             col2imInputBuffer.View,
                                                                                             im2colHeight,
                                                                                             im2colWidth,
                                                                                             0,
                                                                                             output_offset,
                                                                                             0);

                    // TODO: Consider the possibility of writing derivatives for "data" node right into "data.Derivatives" instead of "partialDerivativeForData". To save RAM.
                    KernelExecutor.Execute(ConvolutionKernels.Col2ImKernel)(new Index1D(col2im_num_kernels),
                                                                            col2imInputBuffer.View,
                                                                            parametersBuffer.View,
                                                                            partialDerivativeForData.View,
                                                                            data_im_offset);
                }

                data_im_offset += inputImageLength;
                data_col_offset += im2colLength;
                output_offset += outputImageLength;
            }

            if (!(data is Placeholder))
            {
                KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(data.Derivatives.Length),
                                                               data.Derivatives.View,
                                                               partialDerivativeForData.View);
            }
        }
    }
}
