using ILGPU;
using ILGPU.Runtime;
using System;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class BatchNorm2D : Operation
    {
        private readonly Node data,
                              gamma,
                              beta;

        private readonly Tensor partialDerivativeForData,
                                partialDerivativeForGamma,
                                partialDerivativeForBeta;

        private readonly int batchSize,
                             channelsCount,
                             channelLength;

        internal readonly float Epsilon,
                                Momentum;

        private readonly MemoryBuffer1D<float, Stride1D.Dense> meanBuffer,
                                                               storedMeanBuffer,
                                                               varianceBuffer,
                                                               storedVarianceBuffer,
                                                               normalizedBuffer,
                                                               outputDeltaBuffer,
                                                               meanDeltaBuffer,
                                                               varianceDeltaBuffer;

        internal BatchNorm2D(Graph graph,
                             Node data, Node gamma, Node beta,
                             float epsilon = 1e-05f, float momentum = 0.1f) : base(graph, data, gamma, beta)
        {
            if (data.Output.GetShape().Length != 4)
            {
                throw new Exception("Input should be of (N,C,H,W) shape.");
            }
            if (gamma.Output.GetShape().Length != 1 || gamma.Output.Length != data.Output.GetShape(1))
            {
                throw new Exception("Gamma should be a one dimensional tensor with length equal to number of input channels.");
            }
            if (beta.Output.GetShape().Length != 1 || beta.Output.Length != data.Output.GetShape(1))
            {
                throw new Exception("Beta should be a one dimensional tensor with length equal to number of input channels.");
            }

            this.data = data;
            this.gamma = gamma;
            this.beta = beta;
            this.Epsilon = epsilon;
            this.Momentum = momentum;

            Output = new Tensor(this, data.Output.GetShape());
            Derivatives = new Tensor(this, Output.GetShape());
            if (!(data is Placeholder))
            {
                partialDerivativeForData = new Tensor(this, data.Derivatives.GetShape());
            }
            partialDerivativeForGamma = new Tensor(this, gamma.Derivatives.GetShape());
            partialDerivativeForBeta = new Tensor(this, beta.Derivatives.GetShape());

            batchSize = data.Output.GetShape(0);
            channelsCount = data.Output.GetShape(1);
            channelLength = data.Output.GetShape(2) * data.Output.GetShape(3);

            meanBuffer = KernelExecutor.Accelerator.Allocate1D<float>(channelsCount);
            storedMeanBuffer = KernelExecutor.Accelerator.Allocate1D<float>(channelsCount);
            varianceBuffer = KernelExecutor.Accelerator.Allocate1D<float>(channelsCount);
            storedVarianceBuffer = KernelExecutor.Accelerator.Allocate1D<float>(channelsCount);
            normalizedBuffer = KernelExecutor.Accelerator.Allocate1D<float>(Output.Length);
            outputDeltaBuffer = KernelExecutor.Accelerator.Allocate1D<float>(Output.Length);
            meanDeltaBuffer = KernelExecutor.Accelerator.Allocate1D<float>(channelsCount);
            varianceDeltaBuffer = KernelExecutor.Accelerator.Allocate1D<float>(channelsCount);

            storedMeanBuffer.MemSetToZero();
            storedVarianceBuffer.MemSetToZero();
        }

        internal override void Forward()
        {
            meanBuffer.MemSetToZero();
            varianceBuffer.MemSetToZero();
            normalizedBuffer.MemSetToZero();
            outputDeltaBuffer.MemSetToZero();
            meanDeltaBuffer.MemSetToZero();
            varianceDeltaBuffer.MemSetToZero();

            if (graph.IsTraining)
            {
                KernelExecutor.Execute(BatchNorm2DKernels.MeanForwardKernel)(new Index1D(channelsCount),
                                                                             data.Output.View,
                                                                             meanBuffer.View,
                                                                             storedMeanBuffer.View,
                                                                             Momentum,
                                                                             batchSize,
                                                                             channelsCount,
                                                                             channelLength);

                KernelExecutor.Execute(BatchNorm2DKernels.VarianceForwardKernel)(new Index1D(channelsCount),
                                                                                 data.Output.View,
                                                                                 meanBuffer.View,
                                                                                 varianceBuffer.View,
                                                                                 storedVarianceBuffer.View,
                                                                                 Momentum,
                                                                                 batchSize,
                                                                                 channelsCount,
                                                                                 channelLength);

                KernelExecutor.Execute(BatchNorm2DKernels.NormalizeForwardKernel)(new Index1D(Output.Length),
                                                                                  data.Output.View,
                                                                                  meanBuffer.View,
                                                                                  varianceBuffer.View,
                                                                                  normalizedBuffer.View,
                                                                                  Epsilon,
                                                                                  channelsCount,
                                                                                  channelLength);
            }
            else
            {
                KernelExecutor.Execute(BatchNorm2DKernels.NormalizeForwardKernel)(new Index1D(Output.Length),
                                                                                  data.Output.View,
                                                                                  storedMeanBuffer.View,
                                                                                  storedVarianceBuffer.View,
                                                                                  normalizedBuffer.View,
                                                                                  Epsilon,
                                                                                  channelsCount,
                                                                                  channelLength);
            }

            KernelExecutor.Execute(BatchNorm2DKernels.OutputForwardKernel)(new Index1D(Output.Length),
                                                                           normalizedBuffer.View,
                                                                           gamma.Output.View,
                                                                           beta.Output.View,
                                                                           Output.View,
                                                                           channelsCount,
                                                                           channelLength);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(BatchNorm2DKernels.OutputBackwardKernel)(new Index1D(Derivatives.Length),
                                                                            Derivatives.View,
                                                                            normalizedBuffer.View,
                                                                            gamma.Output.View,
                                                                            partialDerivativeForGamma.View,
                                                                            partialDerivativeForBeta.View,
                                                                            outputDeltaBuffer.View,
                                                                            channelsCount,
                                                                            channelLength);

            KernelExecutor.Execute(BatchNorm2DKernels.MeanBackwardKernel)(new Index1D(channelsCount),
                                                                          outputDeltaBuffer.View,
                                                                          varianceBuffer.View,
                                                                          meanDeltaBuffer.View,
                                                                          Epsilon,
                                                                          batchSize,
                                                                          channelsCount,
                                                                          channelLength);

            KernelExecutor.Execute(BatchNorm2DKernels.VarianceBackwardKernel)(new Index1D(channelsCount),
                                                                              outputDeltaBuffer.View,
                                                                              data.Output.View,
                                                                              meanBuffer.View,
                                                                              varianceBuffer.View,
                                                                              varianceDeltaBuffer.View,
                                                                              Epsilon,
                                                                              batchSize,
                                                                              channelsCount,
                                                                              channelLength);

            KernelExecutor.Execute(BatchNorm2DKernels.NormalizeBackwardKernel)(new Index1D(Derivatives.Length),
                                                                               outputDeltaBuffer.View,
                                                                               meanDeltaBuffer.View,
                                                                               varianceDeltaBuffer.View,
                                                                               data.Output.View,
                                                                               meanBuffer.View,
                                                                               varianceBuffer.View,
                                                                               partialDerivativeForData.View,
                                                                               Epsilon,
                                                                               batchSize,
                                                                               channelsCount,
                                                                               channelLength);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(data.Derivatives.Length),
                                                           data.Derivatives.View,
                                                           partialDerivativeForData.View);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(gamma.Derivatives.Length),
                                                           gamma.Derivatives.View,
                                                           partialDerivativeForGamma.View);

            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(beta.Derivatives.Length),
                                                           beta.Derivatives.View,
                                                           partialDerivativeForBeta.View);
        }

        internal float[] GetMean()
        {
            return storedMeanBuffer.GetAsArray1D();
        }

        internal float[] GetVariance()
        {
            return storedVarianceBuffer.GetAsArray1D();
        }

        internal void LoadMean(float[] mean)
        {
            storedMeanBuffer.CopyFromCPU(mean);
        }

        internal void LoadVariance(float[] variance)
        {
            storedVarianceBuffer.CopyFromCPU(variance);
        }
    }
}
