using Google.Protobuf.Collections;
using Onnx;
using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Nodes;

namespace Pinops.Core.Serialization
{
    internal static class OnnxExtensionMethods
    {
        internal static bool TryAddTensor(this RepeatedField<TensorProto> tensors, Node input, int[] shape)
        {
            if (input is Variable)
            {
                var tensor = new TensorProto()
                {
                    DataType = 1,
                    Name = input.OnnxId
                };
                for (int i = 0; i < shape.Length; i++)
                {
                    tensor.Dims.Add(shape[i]);
                }
                var weights = input.Output.As1DArray();
                for (int i = 0; i < weights.Length; i++)
                {
                    tensor.FloatData.Add(weights[i]);
                }
                tensors.Add(tensor);

                return true;
            }

            return false;
        }

        internal static bool TryAddInputValueInfo(this RepeatedField<ValueInfoProto> valueInfos, Node input, int[] shape)
        {
            if (input is Placeholder || input is Variable)
            {
                return valueInfos.TryAddValueInfo(input, shape);
            }

            return false;
        }

        internal static bool TryAddOutputValueInfo(this RepeatedField<ValueInfoProto> valueInfos, Node operation, int[] shape)
        {
            if (operation is Operation)
            {
                return valueInfos.TryAddValueInfo(operation, shape);
            }

            return false;
        }

        private static bool TryAddValueInfo(this RepeatedField<ValueInfoProto> valueInfos, Node input, int[] shape)
        {
            try
            {
                var type = new TypeProto()
                {
                    TensorType = new TypeProto.Types.Tensor()
                    {
                        ElemType = 1,
                        Shape = new TensorShapeProto()
                    }
                };
                for (int i = 0; i < shape.Length; i++)
                {
                    type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension() { DimValue = shape[i] });
                }
                var valueInfo = new ValueInfoProto()
                {
                    Name = input.OnnxId,
                    Type = type
                };
                valueInfos.Add(valueInfo);

                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }
    }
}
