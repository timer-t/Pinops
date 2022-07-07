using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core.Serialization.LossOperationParsers
{
    internal class MSEParser : LossOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "MSE";
            }
        }

        protected sealed override LossOperationProto GetProto(Operation operation)
        {
            var mse = (MSE)operation;
            var lossOperationProto = new LossOperationProto()
            {
                Type = mse.GetType().Name,
                ObservedValuesPlaceholderShape = mse.InputNodes[1].GetShape(),
                ObservedValuesPlaceholderOnnxId = mse.InputNodes[1].OnnxId
            };

            return lossOperationProto;
        }

        protected sealed override Operation ParseAndAddToGraph(Graph graph, LossOperationProto lossOperationProto, List<Node> inputNodes)
        {
            return new MSE(graph,
                           inputNodes[0],
                           (Placeholder)inputNodes[1]);
        }
    }
}
