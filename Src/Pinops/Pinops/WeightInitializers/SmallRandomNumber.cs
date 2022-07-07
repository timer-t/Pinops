using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.WeightInitializers
{
    public class SmallRandomNumber : WeightInitializer
    {
        public SmallRandomNumber()
        {

        }

        protected override void Init()
        {

        }

        protected override float GetWeight()
        {
            return (float)Helper.GetSmallRandomNumber();
        }
    }
}
