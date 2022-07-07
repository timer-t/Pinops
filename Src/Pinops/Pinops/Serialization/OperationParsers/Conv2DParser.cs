using Google.Protobuf.Collections;
using Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;

namespace Pinops.Core.Serialization.OperationParsers
{
    internal class Conv2DParser : OnnxOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "Conv2D";
            }
        }

        protected sealed override string OnnxOpType
        {
            get
            {
                return "Conv";
            }
        }

        protected sealed override (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation)
        {
            var conv2D = (Conv2D)operation;

            var data = conv2D.InputNodes[0];
            var weights = conv2D.InputNodes[1];

            var node = new NodeProto()
            {
                OpType = OnnxOpType,
                Name = conv2D.OnnxId
            };

            // Attributes.
            var auto_pad = new AttributeProto()
            {
                Name = "auto_pad",
                Type = AttributeProto.Types.AttributeType.String,
                S = Google.Protobuf.ByteString.CopyFromUtf8("NOTSET")
            };
            node.Attribute.Add(auto_pad);
            var dilations = new AttributeProto()
            {
                Name = "dilations",
                Type = AttributeProto.Types.AttributeType.Ints
            };
            dilations.Ints.Add(conv2D.Dilation);
            dilations.Ints.Add(conv2D.Dilation);
            node.Attribute.Add(dilations);
            var group = new AttributeProto()
            {
                Name = "group",
                Type = AttributeProto.Types.AttributeType.Int,
                I = 1
            };
            node.Attribute.Add(group);
            var kernel_shape = new AttributeProto()
            {
                Name = "kernel_shape",
                Type = AttributeProto.Types.AttributeType.Ints
            };
            kernel_shape.Ints.Add(conv2D.Kernel);
            kernel_shape.Ints.Add(conv2D.Kernel);
            node.Attribute.Add(kernel_shape);
            var pads = new AttributeProto()
            {
                Name = "pads",
                Type = AttributeProto.Types.AttributeType.Ints
            };
            pads.Ints.Add(conv2D.Padding);
            pads.Ints.Add(conv2D.Padding);
            pads.Ints.Add(conv2D.Padding);
            pads.Ints.Add(conv2D.Padding);
            node.Attribute.Add(pads);
            var strides = new AttributeProto()
            {
                Name = "strides",
                Type = AttributeProto.Types.AttributeType.Ints
            };
            strides.Ints.Add(conv2D.Stride);
            strides.Ints.Add(conv2D.Stride);
            node.Attribute.Add(strides);

            node.Input.Add(data.OnnxId);
            node.Input.Add(weights.OnnxId);
            node.Output.Add(conv2D.OnnxId);

            RepeatedField<TensorProto> initializer = new RepeatedField<TensorProto>(); // Weights.
            RepeatedField<ValueInfoProto> input = new RepeatedField<ValueInfoProto>(); // Operation input.
            RepeatedField<ValueInfoProto> valueInfo = new RepeatedField<ValueInfoProto>(); // Operation output.

            // Weights. Shape: (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps.
            initializer.TryAddTensor(weights, new int[] { conv2D.Channels, data.GetShape(1), conv2D.Kernel, conv2D.Kernel });
            input.TryAddInputValueInfo(weights, new int[] { conv2D.Channels, data.GetShape(1), conv2D.Kernel, conv2D.Kernel });

            // Input. Shape: (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image.
            input.TryAddInputValueInfo(data, data.GetShape());

            // Output.
            valueInfo.TryAddOutputValueInfo(conv2D, conv2D.GetShape());

            return (node, initializer, input, valueInfo);
        }

        protected sealed override void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            if (inputNodes[1].GetShape(2) != inputNodes[1].GetShape(3))
            {
                throw new Exception("Non equal width and height of kernel are not supported.");
            }

            var currentLayerChannels = inputNodes[1].GetShape(0);
            var previousLayerChannels = inputNodes[1].GetShape(1);
            var kernel = inputNodes[1].GetShape(2);

            inputNodes[1].Output.SetShape(new int[]
            {
                currentLayerChannels,
                previousLayerChannels * kernel * kernel
            });

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
            if (dilations == null)
            {
                throw new Exception("Dilations attribute not found.");
            }
            if (dilations.Ints.Any(i => i != dilations.Ints[0]))
            {
                throw new Exception("Unequal dilation are not supported.");
            }

            var op = graph.Conv2D(inputNodes[0],
                                  (Variable)inputNodes[1],
                                  currentLayerChannels,
                                  kernel,
                                  (int)strides.Ints[0],
                                  (int)pads.Ints[0],
                                  (int)dilations.Ints[0]);
            op.OnnxId = nodeProto.Name;
        }
    }
}
