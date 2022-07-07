using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Pinops.Core.Optimizers;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core.Serialization.OptimizerParsers
{
    internal class AdamParser : OptimizerParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "Adam";
            }
        }

        protected sealed override OptimizerProto ParseAndGetProto(Optimizer optimizer)
        {
            var adam = (Adam)optimizer;

            return new OptimizerProto()
            {
                Type = OpType,
                LearningRate = adam.LearningRate,
                Beta1 = adam.Beta1,
                Beta2 = adam.Beta2,
                Epsilon = adam.Epsilon,
                T = adam.T,
                MAM = adam.Variables.Select(v => new MAM(v.MAM.As1DArray())).ToList(),
                MAV = adam.Variables.Select(v => new MAV(v.MAV.As1DArray())).ToList()
            };
        }

        protected sealed override Optimizer ParseAndGetOptimizer(OptimizerProto optimizerProto)
        {
            return new Adam(optimizerProto.LearningRate, optimizerProto.Beta1, optimizerProto.Beta2, optimizerProto.Epsilon)
            {
                T = optimizerProto.T,
                MAM = optimizerProto.MAM.Select(x => x.Value).ToList(),
                MAV = optimizerProto.MAV.Select(x => x.Value).ToList()
            };
        }
    }
}
