using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Enums;

namespace Pinops.Core.WeightInitializers
{
    public class KaimingHeNormal : WeightInitializer
    {
        private Normal normalDistribution;

        private double sd;

        public KaimingHeNormal(float gain = 1f, FanMode fanMode = FanMode.FanIn)
        {
            this.gain = gain;
            this.fanMode = fanMode;
        }

        protected override void Init()
        {
            if (fanIn == 0d || fanOut == 0d)
            {
                throw new Exception(@"""fanIn"" or ""fanOut"" field are not set.");
            }

            fanValue = fanMode == FanMode.FanIn ? fanIn : fanOut;

            sd = gain / Math.Sqrt(fanValue);
            normalDistribution = new Normal(0d, Math.Pow(sd, 2d));
        }

        protected override float GetWeight()
        {
            return (float)normalDistribution.Sample();
        }
    }
}
