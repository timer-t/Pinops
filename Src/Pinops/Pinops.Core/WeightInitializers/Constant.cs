using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.WeightInitializers
{
    public class Constant : WeightInitializer
    {
        private readonly float value;

        public Constant(float value)
        {
            this.value = value;
        }

        protected override void Init()
        {

        }

        protected override float GetWeight()
        {
            return value;
        }
    }
}
