using System.Linq;

namespace Pinops.Core.Nodes
{
    public abstract class Operation : Node
    {
        internal Node[] InputNodes;

        internal Operation(Graph graph, params Node[] inputNodes) : base(graph)
        {
            InputNodes = inputNodes;
            this.graph?.Operations.Add(this);
        }

        internal abstract void Forward();
        internal abstract void Backward();

        public Tensor Execute()
        {
            foreach (var operation in InputNodes.OfType<Operation>())
            {
                operation.Execute();
            }

            Forward();

            return Output;
        }
    }
}
