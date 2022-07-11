using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Pinops.Core.Optimizers;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core.Serialization.OptimizerParsers
{
    internal abstract class OptimizerParser
    {
        private static readonly List<OptimizerParser> ConcreteParsers = new List<OptimizerParser>();

        static OptimizerParser()
        {
            ConcreteParsers.Add(new SGDParser());
            ConcreteParsers.Add(new AdamParser());
        }

        private static OptimizerParser GetConcreteParser(string opType)
        {
            var concreteParser = ConcreteParsers.FirstOrDefault(p => p.OpType == opType);
            if (concreteParser == null)
            {
                throw new Exception("Concrete parser not found.");
            }
            return concreteParser;
        }

        internal static OptimizerProto GetProto(Optimizer optimizer)
        {
            if (optimizer == null)
            {
                return null;
            }

            var optimizerParser = GetConcreteParser(optimizer.GetType().Name);
            return optimizerParser.ParseAndGetProto(optimizer);
        }

        internal static Optimizer GetOptimizer(OptimizerProto optimizerProto)
        {
            if (optimizerProto == null)
            {
                return null;
            }

            var optimizerParser = GetConcreteParser(optimizerProto.Type);
            return optimizerParser.ParseAndGetOptimizer(optimizerProto);
        }

        protected abstract string OpType { get; }

        protected abstract OptimizerProto ParseAndGetProto(Optimizer optimizer);

        protected abstract Optimizer ParseAndGetOptimizer(OptimizerProto optimizerProto);
    }
}
