using Google.Protobuf.Collections;
using Onnx;
using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;

namespace Pinops.Core.Serialization.OperationParsers
{
    internal class AddParser : OnnxOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "Add";
            }
        }

        protected sealed override string OnnxOpType
        {
            get
            {
                return "Add";
            }
        }

        protected sealed override (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation)
        {
            var add = (Add)operation;

            var inputNodeA = add.InputNodes[0];
            var inputNodeB = add.InputNodes[1];

            var node = new NodeProto()
            {
                OpType = OnnxOpType,
                Name = add.OnnxId
            };

            node.Input.Add(inputNodeA.OnnxId);
            node.Input.Add(inputNodeB.OnnxId);
            node.Output.Add(add.OnnxId);

            RepeatedField<TensorProto> initializer = new RepeatedField<TensorProto>(); // Weights.
            RepeatedField<ValueInfoProto> input = new RepeatedField<ValueInfoProto>(); // Operation input.
            RepeatedField<ValueInfoProto> valueInfo = new RepeatedField<ValueInfoProto>(); // Operation output.

            // Weights.
            initializer.TryAddTensor(inputNodeA, inputNodeA.GetShape());
            initializer.TryAddTensor(inputNodeB, inputNodeB.GetShape());

            // Input, Output.
            input.TryAddInputValueInfo(inputNodeA, inputNodeA.GetShape());
            input.TryAddInputValueInfo(inputNodeB, inputNodeB.GetShape());
            valueInfo.TryAddOutputValueInfo(add, add.GetShape());

            return (node, initializer, input, valueInfo);
        }

        protected sealed override void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            var op = graph.Add(inputNodes[0], inputNodes[1]);
            op.OnnxId = nodeProto.Name;
        }
    }
}
