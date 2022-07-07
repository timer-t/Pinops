using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Enums;

namespace Pinops.Core.WeightInitializers
{
    public static class GainCalculator
    {
        public static float CalculateGain(Nonlinearity nonlinearity, float? param = null)
        {
            return nonlinearity switch
            {
                Nonlinearity.Linear => 1f,
                Nonlinearity.Identity => 1f,
                Nonlinearity.Conv => 1f,
                Nonlinearity.Sigmoid => 1f,
                Nonlinearity.Tanh => 5f / 3f,
                Nonlinearity.ReLU => MathF.Sqrt(2),
                Nonlinearity.LeakyRelu => MathF.Sqrt(2 / (1 + MathF.Pow(param ?? DefaultParams.LeakyReluNegativeSlope, 2))),
                Nonlinearity.SELU => 3f / 4f,
                _ => 1f
            };
        }
    }
}
