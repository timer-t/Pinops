using ProtoBuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.Serialization.Models
{
    [ProtoContract]
    internal class LossOperationProto
    {
        [ProtoMember(1)]
        internal string Type { get; set; }
        [ProtoMember(2)]
        internal Dictionary<string, string> Attributes { get; set; } = new Dictionary<string, string>();
        [ProtoMember(3)]
        internal int[] ObservedValuesPlaceholderShape { get; set; }
        [ProtoMember(4)]
        internal string ObservedValuesPlaceholderOnnxId { get; set; }
    }
}
