using Onnx;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Pinops.Core.Nodes.Operations;
using Pinops.Core.Serialization.LossOperationParsers;
using Pinops.Core.Serialization.Models;
using Pinops.Core.Serialization.OptimizerParsers;

namespace Pinops.Core.Serialization
{
    public static class Serializer
    {
        public static async Task SaveCheckpoint(Graph graph, Trainer trainer, string folderPath, string fileName)
        {
            await Task.Run(()=>
            {
                var modelProto = new ModelProto()
                {
                    Graph = OnnxGraphParser.GetProto(graph)
                };

                modelProto.WriteToFile($"{Path.Combine(folderPath, fileName)}.onnx");

                var checkpoint = new Checkpoint()
                {
                    CurrentEpoch = trainer.CurrentEpoch,
                    EpochsCount = trainer.EpochsCount,
                    IsModelTraining = graph.IsTraining,
                    OptimizerProto = OptimizerParser.GetProto(trainer.Optimizer),
                    LossOperationProto = LossOperationParser.GetLossOperationProto(graph.Operations.FirstOrDefault(op => op is ILossOperation)),
                    FeedDict = trainer.FeedDict?.Select(kvp => new KeyValuePair<string, List<FloatData>>(kvp.Key.OnnxId, kvp.Value.Select(x => new FloatData(x)).ToList()))
                                                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
                };

                using var ms = new MemoryStream();

                ProtoBuf.Serializer.Serialize(ms, checkpoint);
                File.WriteAllBytes($"{Path.Combine(folderPath, fileName)}.cp", ms.ToArray());
            });
        }

        public static async Task<(Graph, Trainer)> LoadCheckpoint(string folderPath, string fileName)
        {
            return await Task.Run(() =>
            {
                var modelProto = ModelProto.Parser.ParseFromFile($"{Path.Combine(folderPath, fileName)}.onnx");

                var graph = OnnxGraphParser.ParseProto(modelProto.Graph);

                using var ms = new MemoryStream(File.ReadAllBytes($"{Path.Combine(folderPath, fileName)}.cp"));

                var checkpoint = ProtoBuf.Serializer.Deserialize<Checkpoint>(ms);

                var trainer = new Trainer()
                {
                    CurrentEpoch = checkpoint.CurrentEpoch,
                    EpochsCount = checkpoint.EpochsCount,
                    Optimizer = OptimizerParser.GetOptimizer(checkpoint.OptimizerProto),
                    Checkpoint = checkpoint
                };

                graph.IsTraining = checkpoint.IsModelTraining;

                return (graph, trainer);
            });
        }
    }
}
