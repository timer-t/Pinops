using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core.Serialization.LossOperationParsers
{
    internal abstract class LossOperationParser
    {
        private static readonly List<LossOperationParser> ConcreteParsers = new List<LossOperationParser>();

        static LossOperationParser()
        {
            ConcreteParsers.Add(new YoloV1LossParser());
            ConcreteParsers.Add(new MSEParser());
        }

        private static LossOperationParser GetConcreteParser(string opType)
        {
            var concreteParser = ConcreteParsers.FirstOrDefault(p => p.OpType == opType);
            if (concreteParser == null)
            {
                throw new Exception("Concrete parser not found.");
            }
            return concreteParser;
        }

        internal static LossOperationProto GetLossOperationProto(Operation operation)
        {
            if (operation == null)
            {
                return null;
            }

            var lossOperationParser = GetConcreteParser(operation.GetType().Name);
            return lossOperationParser.GetProto(operation);
        }

        internal static Operation GetLossOperation(Graph graph, LossOperationProto lossOperationProto, List<Node> inputNodes)
        {
            var lossOperationParser = GetConcreteParser(lossOperationProto.Type);
            return lossOperationParser.ParseAndAddToGraph(graph, lossOperationProto, inputNodes);
        }

        protected abstract string OpType { get; }

        protected abstract LossOperationProto GetProto(Operation operation);

        protected abstract Operation ParseAndAddToGraph(Graph graph, LossOperationProto lossOperationProto, List<Node> inputNodes);
    }
}
