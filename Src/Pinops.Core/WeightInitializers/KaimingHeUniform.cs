using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Enums;

namespace Pinops.Core.WeightInitializers
{
    public class KaimingHeUniform : WeightInitializer
    {
        private ContinuousUniform uniformDistribution;

        private double sd;

        public KaimingHeUniform(float gain = 1f, FanMode fanMode = FanMode.FanIn)
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

            sd = gain * Math.Sqrt(3d / fanValue);
            uniformDistribution = new ContinuousUniform(-sd, sd);
        }

        protected override float GetWeight()
        {
            return (float)uniformDistribution.Sample();
        }
    }
}
