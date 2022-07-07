namespace Pinops.Core.Nodes
{
    public class Placeholder : Node
    {
        internal Placeholder(Graph graph) : base(graph)
        {
            this.graph.Placeholders.Add(this);
        }
        internal Placeholder(Graph graph, float[,] matrix) : this(graph)
        {
            Output = new Tensor(this, matrix);
        }
        internal Placeholder(Graph graph, int[] shape) : this(graph)
        {
            Output = new Tensor(this, shape);
        }

        internal void Load(float[,] matrix)
        {
            Output.Load(matrix);
        }
        public void Load(float[] data)
        {
            Output.Load(data);
        }
    }
}
