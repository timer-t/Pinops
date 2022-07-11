using ProtoBuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.Serialization.Models
{
    [ProtoContract]
    internal class MAV
    {
        [ProtoMember(1)]
        internal float[] Value { get; set; }

        internal MAV()
        {

        }

        internal MAV(float[] value)
        {
            Value = value;
        }
    }
}
