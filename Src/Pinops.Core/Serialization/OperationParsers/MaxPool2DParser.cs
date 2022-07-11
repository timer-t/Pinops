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
    internal class MaxPool2DParser : OnnxOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "MaxPool2D";
            }
        }

        protected sealed override string OnnxOpType
        {
            get
            {
                return "MaxPool";
            }
        }

        protected sealed override (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation)
        {
            var maxPool2D = (MaxPool2D)operation;

            var data = maxPool2D.InputNodes[0];

            var node = new NodeProto()
            {
                OpType = OnnxOpType,
                Name = maxPool2D.OnnxId
            };

            // Attributes.
            var auto_pad = new AttributeProto()
            {
                Name = "auto_pad",
                Type = AttributeProto.Types.AttributeType.String,
                S = Google.Protobuf.ByteString.CopyFromUtf8("NOTSET")
            };
            node.Attribute.Add(auto_pad);
            var kernel_shape = new AttributeProto()
            {
                Name = "kernel_shape",
                Type = AttributeProto.Types.AttributeType.Ints
            };
            kernel_shape.Ints.Add(maxPool2D.Kernel);
            kernel_shape.Ints.Add(maxPool2D.Kernel);
            node.Attribute.Add(kernel_shape);
            var pads = new AttributeProto()
            {
                Name = "pads",
                Type = AttributeProto.Types.AttributeType.Ints
            };
            pads.Ints.Add(maxPool2D.Padding);
            pads.Ints.Add(maxPool2D.Padding);
            pads.Ints.Add(maxPool2D.Padding);
            pads.Ints.Add(maxPool2D.Padding);
            node.Attribute.Add(pads);
            var strides = new AttributeProto()
            {
                Name = "strides",
                Type = AttributeProto.Types.AttributeType.Ints
            };
            strides.Ints.Add(maxPool2D.Stride);
            strides.Ints.Add(maxPool2D.Stride);
            node.Attribute.Add(strides);

            node.Input.Add(data.OnnxId);
            node.Output.Add(maxPool2D.OnnxId);

            RepeatedField<TensorProto> initializer = new RepeatedField<TensorProto>(); // Weights.
            RepeatedField<ValueInfoProto> input = new RepeatedField<ValueInfoProto>(); // Operation input.
            RepeatedField<ValueInfoProto> valueInfo = new RepeatedField<ValueInfoProto>(); // Operation output.

            // Input. Shape: (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image.
            input.TryAddInputValueInfo(data, data.GetShape());

            // Output.
            valueInfo.TryAddOutputValueInfo(maxPool2D, maxPool2D.GetShape());

            return (node, initializer, input, valueInfo);
        }

        protected sealed override void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            var kernel_shape = nodeProto.Attribute.FirstOrDefault(a => a.Name == "kernel_shape");
            if (kernel_shape == null)
            {
                throw new Exception("Kernel_shape attribute not found.");
            }
            if (kernel_shape.Ints.Any(i => i != kernel_shape.Ints[0]))
            {
                throw new Exception("Unequal kernel shape are not supported.");
            }

            var strides = nodeProto.Attribute.FirstOrDefault(a => a.Name == "strides");
            if (strides == null)
            {
                throw new Exception("Strides attribute not found.");
            }
            if (strides.Ints.Any(i => i != strides.Ints[0]))
            {
                throw new Exception("Unequal strides are not supported.");
            }

            var pads = nodeProto.Attribute.FirstOrDefault(a => a.Name == "pads");
            if (pads == null)
            {
                throw new Exception("Pads attribute not found.");
            }
            if (pads.Ints.Any(i => i != pads.Ints[0]))
            {
                throw new Exception("Unequal padding are not supported.");
            }

            var dilations = nodeProto.Attribute.FirstOrDefault(a => a.Name == "dilations");
            if (dilations != null)
            {
                throw new Exception("Dilations for \"MaxPool2D\" are not supported.");
            }

            var op = graph.MaxPool2D(inputNodes[0],
                                     (int)kernel_shape.Ints[0],
                                     (int)strides.Ints[0],
                                     (int)pads.Ints[0]);
            op.OnnxId = nodeProto.Name;
        }
    }
}
