using Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;
using Pinops.Core.Serialization.OperationParsers;

namespace Pinops.Core.Serialization
{
    internal static class OnnxGraphParser
    {
        static OnnxGraphParser()
        {

        }

        internal static GraphProto GetProto(Graph graph)
        {
            var graphProto = new GraphProto();
            foreach (var op in graph.Operations.Where(op => !(op is ILossOperation)))
            {
                var protos = OnnxOperationParser.GetProto(op);
                if (protos.Item1 != null &&
                    protos.Item2 != null &&
                    protos.Item3 != null &&
                    protos.Item4 != null)
                {
                    graphProto.Node.Add(protos.Item1);
                    graphProto.Initializer.AddRange(protos.Item2);
                    graphProto.Input.AddRange(protos.Item3);
                    graphProto.ValueInfo.AddRange(protos.Item4);
                }
            }

            return graphProto;
        }

        internal static Graph ParseProto(GraphProto graphProto)
        {
            Graph graph = new Graph();

            // Parsing each node.
            foreach (var nodeProto in graphProto.Node)
            {
                var inputNodes = new List<Node>();

                // Collecting input nodes for operation.
                foreach (var inputId in nodeProto.Input)
                {
                    var inputOp = graph.Operations.FirstOrDefault(n => n.OnnxId == inputId);
                    if (inputOp == null)
                    {
                        var initializer = graphProto.Initializer.FirstOrDefault(init => init.Name == inputId);
                        if (initializer == null)
                        {
                            // Input Placeholder.
                            var input = graphProto.Input.FirstOrDefault(input => input.Name == inputId);
                            if (input == null)
                            {
                                throw new Exception("Input placeholder not found.");
                            }
                            var shape = new int[input.Type.TensorType.Shape.Dim.Count];
                            for (int i = 0; i < shape.Length; i++)
                            {
                                shape[i] = (int)input.Type.TensorType.Shape.Dim[i].DimValue;
                            }
                            var placeholder = graph.Placeholder(shape);
                            placeholder.OnnxId = input.Name;
                            inputNodes.Add(placeholder);
                        }
                        // Input Variable.
                        else
                        {
                            var shape = new int[initializer.Dims.Count];
                            for (int i = 0; i < shape.Length; i++)
                            {
                                shape[i] = (int)initializer.Dims[i];
                            }
                            var variable = graph.Variable(shape);
                            variable.OnnxId = initializer.Name;
                            variable.Load(initializer.FloatData.ToArray());
                            inputNodes.Add(variable);
                        }
                    }
                    // Input Operation.
                    else
                    {
                        inputNodes.Add(inputOp);
                    }
                }

                // Creating Operation.
                var inputProtos = graphProto.Input.Where(input => nodeProto.Input.Contains(input.Name)).ToList();
                OnnxOperationParser.AddOperationToGraph(graph, nodeProto, inputNodes, inputProtos);
            }

            return graph;
        }
    }
}
