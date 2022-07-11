using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Enums;
using Pinops.Core.Nodes;

namespace Pinops.Core.WeightInitializers
{
    public abstract class WeightInitializer
    {
        private bool isInit = false;

        protected double fanIn,
                         fanOut,
                         fanValue;

        protected float gain = 1f;

        protected FanMode fanMode = FanMode.FanIn;

        public void FillVariable(Variable variable,
                                 int? fanIn = null,
                                 int? fanOut = null)
        {
            if (!isInit)
            {
                this.fanIn = fanIn ?? 0d;
                this.fanOut = fanOut ?? 0d;

                Init();

                isInit = true;
            }

            float[] data = new float[variable.Output.Length];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = GetWeight();
            }
            variable.Load(data);
        }

        protected abstract float GetWeight();

        protected abstract void Init();
    }
}
