using Google.Protobuf.Collections;
using Onnx;
using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;

namespace Pinops.Core.Serialization.OperationParsers
{
    internal class MatMulParser : OnnxOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "MatMul";
            }
        }

        protected sealed override string OnnxOpType
        {
            get
            {
                return "MatMul";
            }
        }

        protected sealed override (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation)
        {
            var matMul = (MatMul)operation;

            var inputA = matMul.InputNodes[0];
            var inputB = matMul.InputNodes[1];

            var node = new NodeProto()
            {
                OpType = OnnxOpType,
                Name = matMul.OnnxId
            };

            node.Input.Add(inputA.OnnxId);
            node.Input.Add(inputB.OnnxId);
            node.Output.Add(matMul.OnnxId);

            RepeatedField<TensorProto> initializer = new RepeatedField<TensorProto>(); // Weights.
            RepeatedField<ValueInfoProto> input = new RepeatedField<ValueInfoProto>(); // Operation input.
            RepeatedField<ValueInfoProto> valueInfo = new RepeatedField<ValueInfoProto>(); // Operation output.

            // Weights.
            initializer.TryAddTensor(inputA, inputA.GetShape());
            initializer.TryAddTensor(inputB, inputB.GetShape());

            // Input, Output.
            input.TryAddInputValueInfo(inputA, inputA.GetShape());
            input.TryAddInputValueInfo(inputB, inputB.GetShape());
            valueInfo.TryAddOutputValueInfo(matMul, matMul.GetShape());

            return (node, initializer, input, valueInfo);
        }

        protected sealed override void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            var op = graph.MatMul(inputNodes[0], inputNodes[1]);
            op.OnnxId = nodeProto.Name;
        }
    }
}
