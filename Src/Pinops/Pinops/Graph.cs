using System;
using System.Collections.Generic;
using System.Linq;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;

namespace Pinops.Core
{
    public class Graph
    {
        private readonly List<Operation> sortedOperations;

        internal List<Operation> Operations;
        internal List<Placeholder> Placeholders;
        internal List<Variable> Variables;
        internal List<Node> Nodes;

        internal bool IsTraining;

        public Placeholder Input
        {
            get
            {
                return Placeholders.First();
            }
        }
        public Operation Output
        {
            get
            {
                return Operations.Last(op => !(op is ILossOperation));
            }
        }

        public Graph()
        {
            sortedOperations = new List<Operation>();

            Operations = new List<Operation>();
            Placeholders = new List<Placeholder>();
            Variables = new List<Variable>();
            Nodes = new List<Node>();
        }

        internal List<Operation> TopologicalSort(Operation operation)
        {
            sortedOperations.Clear();

            void Recurse(Node node)
            {
                if (node is Operation operation)
                {
                    foreach (var inputNode in operation.InputNodes)
                    {
                        Recurse(inputNode);
                    }
                    sortedOperations.Add(operation);
                }
            }

            Recurse(operation);

            return sortedOperations;
        }

        public Tensor Run()
        {
            return Run(Output);
        }

        public Tensor Run(Operation operation)
        {
            IsTraining = false;

            TopologicalSort(operation);

            foreach (var sortedOperation in sortedOperations)
            {
                sortedOperation.Forward();
            }

            return operation.Output;
        }

        public Variable Variable(float[,] matrix)
        {
            return new Variable(this, matrix);
        }

        public Variable Variable(params int[] shape)
        {
            return new Variable(this, shape);
        }

        public Placeholder Placeholder(float[,] matrix)
        {
            return new Placeholder(this, matrix);
        }

        public Placeholder Placeholder(params int[] shape)
        {
            return new Placeholder(this, shape);
        }

        #region Operations
        public Operation MatMul(Node inputNodeA, Node inputNodeB)
        {
            return new MatMul(this, inputNodeA, inputNodeB);
        }

        public Operation Conv2D(Node data, Variable weights,
                                int channels, int kernel, int stride = 1, int padding = 0, int dilation = 1)
        {
            return new Conv2D(this, data, weights, channels, kernel, stride, padding, dilation);
        }

        public Operation AveragePool2D(Node input,
                                       int kernel, int stride = 1, int padding = 0)
        {
            return new AveragePool2D(this, input, kernel, stride, padding);
        }

        public Operation MaxPool2D(Node input,
                                   int kernel, int stride = 1, int padding = 0)
        {
            return new MaxPool2D(this, input, kernel, stride, padding);
        }

        [Obsolete("Use \"Add\" operation instead.", true)]
        public Operation MatrixAddVectorRowWise(Node matrix, Node vector, int matrixHeight, int matrixWidth)
        {
            return new MatrixAddVectorRowWise(this, matrix, vector, matrixHeight, matrixWidth);
        }

        public Operation Add(Node inputA, Node inputB)
        {
            return new Add(this, inputA, inputB);
        }

        public Operation Sum(Node inputNodeA, Node inputNodeB)
        {
            return new Sum(this, inputNodeA, inputNodeB);
        }

        public Operation LogisticSigmoid(Node inputNode)
        {
            return new LogisticSigmoid(this, inputNode);
        }

        public Operation LeakyReLU(Node inputNode, float negativeSlope = DefaultParams.LeakyReluNegativeSlope)
        {
            return new LeakyReLU(this, inputNode, negativeSlope);
        }

        public Operation MSE(Node predicted, Placeholder observed)
        {
            return new MSE(this, predicted, observed);
        }

        public Operation YoloV1Loss(Node predicted,
                                    Placeholder observed,
                                    int widthCellsCount,
                                    int heightCellsCount,
                                    int classesCount,
                                    int boundingBoxesPerCell)
        {
            return new YoloV1Loss(this,
                                  predicted,
                                  observed,
                                  widthCellsCount,
                                  heightCellsCount,
                                  classesCount,
                                  boundingBoxesPerCell);
        }

        public Operation Reshape(Node input, params int[] shape)
        {
            return new Reshape(this, input, shape);
        }

        public Operation BatchNorm2D(Node data, Node gamma, Node beta,
                                     float epsilon = 1e-05f, float momentum = 0.1f)
        {
            return new BatchNorm2D(this, data, gamma, beta, epsilon, momentum);
        }
        #endregion
    }
}
