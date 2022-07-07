using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.WeightInitializers
{
    public class Zeros : WeightInitializer
    {
        public Zeros()
        {

        }

        protected override void Init()
        {

        }

        protected override float GetWeight()
        {
            return 0f;
        }
    }
}
