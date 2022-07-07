using Onnx;
using ProtoBuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.Serialization.Models
{
    [ProtoContract]
    public class Checkpoint
    {
        [ProtoMember(1)]
        internal int CurrentEpoch { get; set; }
        [ProtoMember(2)]
        internal int EpochsCount { get; set; }
        [ProtoMember(3)]
        internal bool IsModelTraining { get; set; }
        [ProtoMember(4)]
        internal OptimizerProto OptimizerProto { get; set; }
        [ProtoMember(5)]
        internal LossOperationProto LossOperationProto { get; set; }
        [ProtoMember(6)]
        internal Dictionary<string, List<FloatData>> FeedDict { get; set; }
    }
}
