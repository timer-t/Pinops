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
    internal class BatchNorm2DParser : OnnxOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "BatchNorm2D";
            }
        }

        protected sealed override string OnnxOpType
        {
            get
            {
                return "BatchNormalization";
            }
        }

        protected sealed override (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation)
        {
            var batchNorm2D = (BatchNorm2D)operation;

            // Inputs.
            var data = batchNorm2D.InputNodes[0];

            var scale = batchNorm2D.InputNodes[1];
            scale.OnnxId = $"{scale.OnnxId}_Gamma";

            var b = batchNorm2D.InputNodes[2];
            b.OnnxId = $"{scale.OnnxId}_Beta";

            var input_mean = new Variable(batchNorm2D.GetShape(1))
            {
                OnnxId = $"{data.OnnxId}_Mean"
            };
            input_mean.Load(batchNorm2D.GetMean());

            var input_var = new Variable(batchNorm2D.GetShape(1))
            {
                OnnxId = $"{data.OnnxId}_Variance"
            };
            input_var.Load(batchNorm2D.GetVariance());

            var node = new NodeProto()
            {
                OpType = OnnxOpType,
                Name = batchNorm2D.OnnxId
            };

            // Attributes.
            var epsilon = new AttributeProto()
            {
                Name = "epsilon",
                Type = AttributeProto.Types.AttributeType.Float,
                F = batchNorm2D.Epsilon
            };
            node.Attribute.Add(epsilon);
            var momentum = new AttributeProto()
            {
                Name = "momentum",
                Type = AttributeProto.Types.AttributeType.Float,
                F = batchNorm2D.Momentum
            };
            node.Attribute.Add(momentum);

            node.Input.Add(data.OnnxId);
            node.Input.Add(scale.OnnxId);
            node.Input.Add(b.OnnxId);
            node.Input.Add(input_mean.OnnxId);
            node.Input.Add(input_var.OnnxId);
            node.Output.Add(batchNorm2D.OnnxId);

            RepeatedField<TensorProto> initializer = new RepeatedField<TensorProto>(); // Weights.
            RepeatedField<ValueInfoProto> input = new RepeatedField<ValueInfoProto>(); // Operation input.
            RepeatedField<ValueInfoProto> valueInfo = new RepeatedField<ValueInfoProto>(); // Operation output.

            // Weights.
            initializer.TryAddTensor(data, data.GetShape());
            initializer.TryAddTensor(scale, scale.GetShape());
            initializer.TryAddTensor(b, b.GetShape());
            initializer.TryAddTensor(input_mean, input_mean.GetShape());
            initializer.TryAddTensor(input_var, input_var.GetShape());

            // Input, Output.
            input.TryAddInputValueInfo(data, data.GetShape());
            input.TryAddInputValueInfo(scale, scale.GetShape());
            input.TryAddInputValueInfo(b, b.GetShape());
            input.TryAddInputValueInfo(input_mean, input_mean.GetShape());
            input.TryAddInputValueInfo(input_var, input_var.GetShape());
            valueInfo.TryAddOutputValueInfo(batchNorm2D, batchNorm2D.GetShape());

            return (node, initializer, input, valueInfo);
        }

        protected sealed override void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            var epsilon = nodeProto.Attribute.FirstOrDefault(a => a.Name == "epsilon");
            if (epsilon == null)
            {
                throw new Exception("Epsilon attribute not found.");
            }

            var momentum = nodeProto.Attribute.FirstOrDefault(a => a.Name == "momentum");
            if (momentum == null)
            {
                throw new Exception("Momentum  attribute not found.");
            }

            var op = (BatchNorm2D)graph.BatchNorm2D(inputNodes[0], inputNodes[1], inputNodes[2], epsilon.F, momentum.F);
            op.LoadMean(inputNodes[3].Output.As1DArray());
            op.LoadVariance(inputNodes[4].Output.As1DArray());
            op.OnnxId = nodeProto.Name;
        }
    }
}
