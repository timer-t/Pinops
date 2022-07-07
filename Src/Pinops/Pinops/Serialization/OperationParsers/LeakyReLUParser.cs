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
    internal class LeakyReLUParser : OnnxOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "LeakyReLU";
            }
        }

        protected sealed override string OnnxOpType
        {
            get
            {
                return "LeakyRelu";
            }
        }

        protected sealed override (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation)
        {
            var leakyReLU = (LeakyReLU)operation;

            var inputNode = leakyReLU.InputNodes[0];

            var node = new NodeProto()
            {
                OpType = OnnxOpType,
                Name = leakyReLU.OnnxId
            };

            // Attributes.
            var negativeSlope = new AttributeProto()
            {
                Name = "alpha",
                Type = AttributeProto.Types.AttributeType.Float,
                F = leakyReLU.NegativeSlope
            };
            node.Attribute.Add(negativeSlope);

            node.Input.Add(inputNode.OnnxId);
            node.Output.Add(leakyReLU.OnnxId);

            RepeatedField<TensorProto> initializer = new RepeatedField<TensorProto>(); // Weights.
            RepeatedField<ValueInfoProto> input = new RepeatedField<ValueInfoProto>(); // Operation input.
            RepeatedField<ValueInfoProto> valueInfo = new RepeatedField<ValueInfoProto>(); // Operation output.

            // Weights.
            initializer.TryAddTensor(inputNode, inputNode.GetShape());

            // Input, Output.
            input.TryAddInputValueInfo(inputNode, inputNode.GetShape());
            valueInfo.TryAddOutputValueInfo(leakyReLU, leakyReLU.GetShape());

            return (node, initializer, input, valueInfo);
        }

        protected sealed override void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            var alpha = nodeProto.Attribute.FirstOrDefault(a => a.Name == "alpha");
            if (alpha == null)
            {
                throw new Exception("Alpha attribute not found.");
            }

            var negativeSlope = alpha.F;

            var op = graph.LeakyReLU(inputNodes[0], negativeSlope);
            op.OnnxId = nodeProto.Name;
        }
    }
}
