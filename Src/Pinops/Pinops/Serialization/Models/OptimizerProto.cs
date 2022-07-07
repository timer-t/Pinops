using ProtoBuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.Serialization.Models
{
    [ProtoContract]
    internal class OptimizerProto
    {
        [ProtoMember(1)]
        internal string Type { get; set; }
        [ProtoMember(2)]
        internal float LearningRate { get; set; }
        [ProtoMember(3)]
        internal float Beta1 { get; set; }
        [ProtoMember(4)]
        internal float Beta2 { get; set; }
        [ProtoMember(5)]
        internal float Epsilon { get; set; }
        [ProtoMember(6)]
        internal float T { get; set; }
        [ProtoMember(7)]
        internal List<MAM> MAM { get; set; }
        [ProtoMember(8)]
        internal List<MAV> MAV { get; set; }
    }
}
