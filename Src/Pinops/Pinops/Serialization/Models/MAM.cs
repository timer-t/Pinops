using ProtoBuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.Serialization.Models
{
    [ProtoContract]
    internal class MAM
    {
        [ProtoMember(1)]
        internal float[] Value { get; set; }

        internal MAM()
        {

        }

        internal MAM(float[] value)
        {
            Value = value;
        }
    }
}
