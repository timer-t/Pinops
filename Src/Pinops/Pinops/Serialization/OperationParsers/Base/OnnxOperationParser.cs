using Google.Protobuf.Collections;
using Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Pinops.Core.Nodes;

namespace Pinops.Core.Serialization.OperationParsers
{
    internal abstract class OnnxOperationParser
    {
        private static readonly List<OnnxOperationParser> ConcreteParsers = new List<OnnxOperationParser>();

        static OnnxOperationParser()
        {
            ConcreteParsers.Add(new MatMulParser());
            ConcreteParsers.Add(new Conv2DParser());
            ConcreteParsers.Add(new LeakyReLUParser());
            ConcreteParsers.Add(new AddParser());
            ConcreteParsers.Add(new ReshapeParser());
            ConcreteParsers.Add(new BatchNorm2DParser());
            ConcreteParsers.Add(new MaxPool2DParser());
        }

        private static OnnxOperationParser GetConcreteParserByOpType(string opType)
        {
            var concreteParser = ConcreteParsers.FirstOrDefault(p => p.OpType == opType);
            if (concreteParser == null)
            {
                throw new Exception("Concrete parser not found.");
            }
            return concreteParser;
        }

        private static OnnxOperationParser GetConcreteParserByOnnxOpType(string opType)
        {
            var concreteParser = ConcreteParsers.FirstOrDefault(p => p.OnnxOpType == opType);
            if (concreteParser == null)
            {
                throw new Exception("Concrete parser not found.");
            }
            return concreteParser;
        }

        internal static (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) GetProto(Operation operation)
        {
            var operationParser = GetConcreteParserByOpType(operation.GetType().Name);
            return operationParser.ParseAndGetProto(operation);
        }

        internal static void AddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos)
        {
            var operationParser = GetConcreteParserByOnnxOpType(nodeProto.OpType);
            operationParser.ParseAndAddOperationToGraph(graph, nodeProto, inputNodes, inputProtos);
        }

        protected abstract string OpType { get; }

        protected abstract string OnnxOpType { get; }

        protected abstract (NodeProto, RepeatedField<TensorProto>, RepeatedField<ValueInfoProto>, RepeatedField<ValueInfoProto>) ParseAndGetProto(Operation operation);

        protected abstract void ParseAndAddOperationToGraph(Graph graph, NodeProto nodeProto, List<Node> inputNodes, List<ValueInfoProto> inputProtos);
    }
}
