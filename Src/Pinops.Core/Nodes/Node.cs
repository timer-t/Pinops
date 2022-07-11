using System;
using System.Diagnostics;
using Pinops.Core.Interfaces;

namespace Pinops.Core.Nodes
{
    [DebuggerDisplay("Type = {GetType().Name}, Shape = {GetShapeString()}")]
    public abstract class Node : ITensorNodeProvider
    {
        internal string OnnxId;
        protected Graph graph;
        internal Tensor Output,
                        Derivatives;

        internal Node(Graph graph)
        {
            if (graph == null)
            {
                return;
            }

            this.graph = graph;
            this.graph.Nodes.Add(this);

            var type = GetType();
            if (type == typeof(Placeholder) || type == typeof(Variable))
            {
                OnnxId = $"{type.Name}_{this.graph.Nodes.Count}_Input";
            }
            else if (type.BaseType == typeof(Operation))
            {
                OnnxId = $"{type.Name}_{this.graph.Nodes.Count}_Output";
            }
        }

        public Tensor GetTensor()
        {
            return Output;
        }
        public Node GetNode()
        {
            return this;
        }

        public int[] GetShape()
        {
            return Output.GetShape();
        }
        public int GetShape(int index)
        {
            return Output.GetShape(index);
        }
        private string GetShapeString()
        {
            return String.Join(", ", Output.GetShape());
        }
        public int GetLength()
        {
            return Output.Length;
        }
    }
}
