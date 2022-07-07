namespace Pinops.Core.Nodes
{
    public class Variable : Node
    {
        internal Tensor MAM,
                        MAV;

        internal Variable(Graph graph) : base(graph)
        {
            if (this.graph == null)
            {
                return;
            }

            this.graph.Variables.Add(this);
        }
        internal Variable(Graph graph, float[,] matrix) : this(graph)
        {
            Output = new Tensor(this, matrix);
            Derivatives = new Tensor(this, Output.GetShape());
        }
        internal Variable(Graph graph, int[] shape) : this(graph)
        {
            Output = new Tensor(this, shape);
            Derivatives = new Tensor(this, Output.GetShape());
        }
        internal Variable(params int[] shape) : this(null, shape)
        {

        }

        public static Variable New(params int[] shape)
        {
            return new Variable(null, shape);
        }
        public static Variable New(float[] data)
        {
            var newVar = new Variable(data.Length);
            newVar.Load(data);
            return newVar;
        }

        public void Load(float[] data)
        {
            Output.Load(data);
        }
        public void Load(float[,] matrix)
        {
            Output.Load(matrix);
        }
    }
}
