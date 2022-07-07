using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.WeightInitializers
{
    public class XavierGlorotNormal : WeightInitializer
    {
        private Normal normalDistribution;

        private double sd;

        public XavierGlorotNormal(float gain = 1f)
        {
            this.gain = gain;
        }

        protected override void Init()
        {
            if (fanIn == 0d || fanOut == 0d)
            {
                throw new Exception(@"""fanIn"" or ""fanOut"" field are not set.");
            }

            sd = gain * Math.Sqrt(2d / (fanIn + fanOut));
            normalDistribution = new Normal(0d, Math.Pow(sd, 2d));
        }

        protected override float GetWeight()
        {
            return (float)normalDistribution.Sample();
        }
    }
}
