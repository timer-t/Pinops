using ILGPU;
using System;
using Pinops.Core.ComputationalKernels;

namespace Pinops.Core.Nodes.Operations
{
    internal class YoloV1Loss : Operation, ILossOperation
    {
        private const float object_scale = 1f,
                            noobject_scale = 0.5f,
                            class_scale = 1f,
                            coord_scale = 5f;

        private readonly Node predicted,
                              observed;

        internal readonly int WidthCellsCount,
                              HeightCellsCount,
                              ClassesCount,
                              BoundingBoxesPerCell,
                              TotalCellsCount,
                              BatchSize;

        private readonly Tensor partialDerivativeForPredicted;

        internal YoloV1Loss(Graph graph,
                            Node predicted,
                            Placeholder observed,
                            int widthCellsCount,
                            int heightCellsCount,
                            int classesCount,
                            int boundingBoxesPerCell) : base(graph, predicted, observed)
        {
            this.predicted = predicted;
            this.observed = observed;
            this.WidthCellsCount = widthCellsCount;
            this.HeightCellsCount = heightCellsCount;
            this.ClassesCount = classesCount;
            this.BoundingBoxesPerCell = boundingBoxesPerCell;
            this.TotalCellsCount = widthCellsCount * heightCellsCount;
            this.BatchSize = predicted.Output.GetShape(0);

            Output = new Tensor(this, BatchSize);
            partialDerivativeForPredicted = new Tensor(this, predicted.Derivatives.GetShape());
        }

        internal override void Forward()
        {
            var err = new float[BatchSize];

            var predictedData = predicted.Output.As1DArray();
            var observedData = observed.Output.As1DArray();

            var delta = new float[predicted.Derivatives.Length];

            var labelLengthPredicted = ClassesCount + (5 * BoundingBoxesPerCell);
            var labelLengthObserved = ClassesCount + 5;

            for (int b = 0; b < BatchSize; b++)
            {
                for (int c = 0; c < TotalCellsCount; c++)
                {
                    var indexPredicted = (b * TotalCellsCount + c) * labelLengthPredicted;
                    var indexObserved = (b * TotalCellsCount + c) * labelLengthObserved;
                    var isObj = observedData[indexObserved + ClassesCount] == 1;
                    if (isObj)
                    {
                        var best_iou = float.MinValue;
                        var bestBoxIndex = 0;
                        var bestBoxMidX = 0f;
                        var bestBoxMidY = 0f;
                        var bestBoxW = 0f;
                        var bestBoxH = 0f;

                        var predictedBoxIndex = indexPredicted + ClassesCount;
                        var observedBoxIndex = indexObserved + ClassesCount;

                        float observedBoxMidX = observedData[observedBoxIndex + 1];
                        float observedBoxMidY = observedData[observedBoxIndex + 2];
                        float observedBoxW = observedData[observedBoxIndex + 3];
                        float observedBoxH = observedData[observedBoxIndex + 4];

                        for (int box = 0; box < BoundingBoxesPerCell; box++)
                        {
                            float predictedBoxMidX = predictedData[predictedBoxIndex + 1];
                            float predictedBoxMidY = predictedData[predictedBoxIndex + 2];
                            float predictedBoxW = predictedData[predictedBoxIndex + 3];
                            float predictedBoxH = predictedData[predictedBoxIndex + 4];

                            var iou = Helper.IoU(predictedBoxMidX,
                                                 predictedBoxMidY,
                                                 predictedBoxW,
                                                 predictedBoxH,
                                                 observedBoxMidX,
                                                 observedBoxMidY,
                                                 observedBoxW,
                                                 observedBoxH);

                            if (iou > best_iou)
                            {
                                best_iou = iou;
                                bestBoxIndex = predictedBoxIndex;
                                bestBoxMidX = predictedBoxMidX;
                                bestBoxMidY = predictedBoxMidY;
                                bestBoxW = predictedBoxW;
                                bestBoxH = predictedBoxH;
                            }

                            predictedBoxIndex += 5;
                        }

                        // Coordinates
                        err[b] += coord_scale * (MathF.Pow(observedBoxMidX - bestBoxMidX, 2) +
                                                 MathF.Pow(observedBoxMidY - bestBoxMidY, 2));
                        err[b] += coord_scale * (MathF.Pow(MathF.Sqrt(observedBoxW) - MathF.Sqrt(MathF.Abs(bestBoxW)), 2) +
                                                 MathF.Pow(MathF.Sqrt(observedBoxH) - MathF.Sqrt(MathF.Abs(bestBoxH)), 2));

                        delta[bestBoxIndex + 1] = coord_scale * 2 * (bestBoxMidX - observedBoxMidX);
                        delta[bestBoxIndex + 2] = coord_scale * 2 * (bestBoxMidY - observedBoxMidY);
                        delta[bestBoxIndex + 3] = coord_scale * ((MathF.Abs(bestBoxW) - MathF.Sqrt(observedBoxW) * MathF.Sqrt(MathF.Abs(bestBoxW))) / bestBoxW);
                        delta[bestBoxIndex + 4] = coord_scale * ((MathF.Abs(bestBoxH) - MathF.Sqrt(observedBoxH) * MathF.Sqrt(MathF.Abs(bestBoxH))) / bestBoxH);

                        // Probability
                        err[b] += object_scale * MathF.Pow(1f - predictedData[bestBoxIndex], 2);

                        delta[bestBoxIndex] = object_scale * 2 * (predictedData[bestBoxIndex] - 1f);

                        // Classes
                        for (int cl = 0; cl < ClassesCount; cl++)
                        {
                            err[b] += class_scale * MathF.Pow(observedData[indexObserved + cl] - predictedData[indexPredicted + cl], 2);

                            delta[indexPredicted + cl] = class_scale * 2 * (predictedData[indexPredicted + cl] - observedData[indexObserved + cl]);
                        }
                    }
                    else
                    {
                        // No object
                        var probabilityIndex = indexPredicted + ClassesCount;
                        for (int box = 0; box < BoundingBoxesPerCell; box++)
                        {
                            err[b] += noobject_scale * MathF.Pow(0f - predictedData[probabilityIndex], 2);

                            delta[probabilityIndex] = noobject_scale * 2 * predictedData[probabilityIndex];

                            probabilityIndex += 5;
                        }
                    }
                }
            }

            Output.Load(err);
            partialDerivativeForPredicted.Load(delta);
        }

        internal override void Backward()
        {
            KernelExecutor.Execute(BasicKernels.AddKernel)(new Index1D(predicted.Derivatives.Length),
                                                           predicted.Derivatives.View,
                                                           partialDerivativeForPredicted.View);
        }
    }
}
