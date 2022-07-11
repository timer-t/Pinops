using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Optimizers;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core.Serialization.OptimizerParsers
{
    internal class SGDParser : OptimizerParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "SGD";
            }
        }

        protected sealed override OptimizerProto ParseAndGetProto(Optimizer optimizer)
        {
            var sgd = (SGD)optimizer;

            return new OptimizerProto()
            {
                Type = OpType,
                LearningRate = sgd.LearningRate
            };
        }

        protected sealed override Optimizer ParseAndGetOptimizer(OptimizerProto optimizerProto)
        {
            return new SGD(optimizerProto.LearningRate);
        }
    }
}
