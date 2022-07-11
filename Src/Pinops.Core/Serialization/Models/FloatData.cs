using ProtoBuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.Serialization.Models
{
    [ProtoContract]
    internal class FloatData
    {
        [ProtoMember(1)]
        internal float[] Value { get; set; }

        internal FloatData()
        {
            
        }

        internal FloatData(float[] value)
        {
            Value = value;
        }
    }
}
