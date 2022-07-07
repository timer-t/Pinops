using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.WeightInitializers
{
    public class XavierGlorotUniform : WeightInitializer
    {
        private ContinuousUniform uniformDistribution;

        private double sd;

        public XavierGlorotUniform(float gain = 1f)
        {
            this.gain = gain;
        }

        protected override void Init()
        {
            if (fanIn == 0d || fanOut == 0d)
            {
                throw new Exception(@"""fanIn"" or ""fanOut"" field are not set.");
            }

            sd = gain * Math.Sqrt(6d / (fanIn + fanOut));
            uniformDistribution = new ContinuousUniform(-sd, sd);
        }

        protected override float GetWeight()
        {
            return (float)uniformDistribution.Sample();
        }
    }
}
