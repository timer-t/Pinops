using Google.Protobuf.Collections;
using Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;

namespace Pinops.Core.Serialization.OperationParsers
{
    internal class ReshapeParser : OnnxOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "Reshape";
            }
        }

        protected sealed override string OnnxOpType
        {
            get
            {
                return "Reshape";
            }
        }

        protected sealed override (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation)
        {
            var reshape = (Reshape)operation;

            var inputNodeA = reshape.InputNodes[0];
            var inputNodeB = new Variable(reshape.Shape.Length)
            {
                OnnxId = $"{inputNodeA.OnnxId}_NewShape"
            };
            inputNodeB.Load(reshape.Shape.Select(x => (float)x).ToArray());

            var node = new NodeProto()
            {
                OpType = OnnxOpType,
                Name = reshape.OnnxId
            };

            node.Input.Add(inputNodeA.OnnxId);
            node.Input.Add(inputNodeB.OnnxId);
            node.Output.Add(reshape.OnnxId);

            RepeatedField<TensorProto> initializer = new RepeatedField<TensorProto>(); // Weights.
            RepeatedField<ValueInfoProto> input = new RepeatedField<ValueInfoProto>(); // Operation input.
            RepeatedField<ValueInfoProto> valueInfo = new RepeatedField<ValueInfoProto>(); // Operation output.

            // Weights.
            initializer.TryAddTensor(inputNodeA, inputNodeA.GetShape());
            initializer.TryAddTensor(inputNodeB, inputNodeB.GetShape());

            // Input, Output.
            input.TryAddInputValueInfo(inputNodeA, inputNodeA.GetShape());
            input.TryAddInputValueInfo(inputNodeB, inputNodeB.GetShape());
            valueInfo.TryAddOutputValueInfo(reshape, reshape.GetShape());

            return (node, initializer, input, valueInfo);
        }

        protected sealed override void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            var op = graph.Reshape(inputNodes[0], inputNodes[1].Output.As1DArray().Select(x => (int)x).ToArray());
            op.OnnxId = nodeProto.Name;
        }
    }
}
