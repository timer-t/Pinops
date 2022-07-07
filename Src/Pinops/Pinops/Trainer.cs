using Nito.AsyncEx;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Optimizers;
using Pinops.Core.Serialization.LossOperationParsers;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core
{
    public class Trainer
    {
        internal int CurrentEpoch { get; set; }
        internal int EpochsCount { get; set; }
        public Optimizer Optimizer { get; set; }
        internal Dictionary<Placeholder, List<float[]>> FeedDict { get; set; }
        internal Checkpoint Checkpoint { get; set; }

        public readonly PauseTokenSource PauseSource;

        public Trainer()
        {
            PauseSource = new PauseTokenSource();
        }

        public void ContinueMinimize(Graph graph)
        {
            var lossOperationProto = Checkpoint.LossOperationProto;

            var observedValuesPlaceholder = graph.Placeholder(lossOperationProto.ObservedValuesPlaceholderShape);
            observedValuesPlaceholder.OnnxId = lossOperationProto.ObservedValuesPlaceholderOnnxId;

            var lossOp = LossOperationParser.GetLossOperation(graph, lossOperationProto, new List<Node>
            {
                graph.Output,
                observedValuesPlaceholder
            });

            var feedDict = Checkpoint.FeedDict.Select(kvp => new KeyValuePair<Placeholder, List<float[]>>(graph.Placeholders.FirstOrDefault(p => p.OnnxId == kvp.Key), kvp.Value.Select(x => x.Value).ToList()))
                                              .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

            Minimize(graph, lossOp, Checkpoint.EpochsCount, feedDict, Checkpoint.CurrentEpoch);
        }

        public void Minimize(Graph graph, Operation lossOperation, int epochs, Dictionary<Placeholder, List<float[]>> feedDict, int currentEpoch = 0)
        {
            graph.IsTraining = true;

            EpochsCount = epochs;
            FeedDict = feedDict;

            Optimizer.Init(graph.Variables);

            var sortedOperations = graph.TopologicalSort(lossOperation);

            var batchesCount = feedDict.First().Value.Count;

            for (int epochIndex = currentEpoch; epochIndex < epochs; epochIndex++)
            {
                CurrentEpoch = epochIndex;

                if (PauseSource.IsPaused)
                {
                    OnGraphMinimizePaused();
                    PauseSource.Token.WaitWhilePaused();
                }

                for (int batchIndex = 0; batchIndex < batchesCount; batchIndex++)
                {
                    foreach (var kvp in feedDict)
                    {
                        kvp.Key.Load(kvp.Value[batchIndex]);
                    }

                    foreach (var sortedOperation in sortedOperations)
                    {
                        sortedOperation.Forward();
                    }
                    foreach (var op in graph.Operations)
                    {
                        if (op.Derivatives != null)
                        {
                            op.Derivatives.Clear();
                        }
                    }
                    foreach (var v in graph.Variables)
                    {
                        if (v.Derivatives != null)
                        {
                            v.Derivatives.Clear();
                        }
                    }
                    for (int i = sortedOperations.Count - 1; i >= 0; i--)
                    {
                        sortedOperations[i].Backward();
                    }

                    Optimizer.Minimize(graph.Variables);
                }

                Console.WriteLine("-------------------------------");
                Console.WriteLine($"Epoch: {epochIndex + 1}");

                var loss = lossOperation.Output.As1DArray();
                var totalError = Decimal.Parse(loss.Average().ToString(), NumberStyles.AllowExponent | NumberStyles.AllowDecimalPoint);
                Console.WriteLine($"Error: {totalError}");

                OnGraphMinimizeUpdated(new GraphMinimizeUpdatedEventArgs(epochIndex + 1, totalError));
            }
        }

        // Events.
        public event EventHandler<GraphMinimizeUpdatedEventArgs> GraphMinimizeUpdated;
        protected virtual void OnGraphMinimizeUpdated(GraphMinimizeUpdatedEventArgs e)
        {
            EventHandler<GraphMinimizeUpdatedEventArgs> handler = GraphMinimizeUpdated;
            handler?.Invoke(this, e);
        }

        public event EventHandler GraphMinimizePaused;
        protected virtual void OnGraphMinimizePaused()
        {
            EventHandler handler = GraphMinimizePaused;
            handler?.Invoke(this, EventArgs.Empty);
        }
    }

    // Events Args.
    public class GraphMinimizeUpdatedEventArgs : EventArgs
    {
        public int Epoch { get; set; }
        public decimal TotalError { get; set; }

        internal GraphMinimizeUpdatedEventArgs(int epoch, decimal totalError)
        {
            Epoch = epoch;
            TotalError = totalError;
        }
    }
}
