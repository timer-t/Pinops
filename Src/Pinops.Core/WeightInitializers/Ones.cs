using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.WeightInitializers
{
    public class Ones : WeightInitializer
    {
        public Ones()
        {

        }

        protected override void Init()
        {

        }

        protected override float GetWeight()
        {
            return 1f;
        }
    }
}
