namespace Pinops.Core.Nodes.Operations
{
    internal class Reshape : Operation
    {
        private readonly Node input;

        private readonly int[] oldShape,
                               newShape;

        internal int[] Shape { get; private set; }

        internal Reshape(Graph graph, Node input, int[] shape) : base(graph, input)
        {
            this.input = input;
            this.Shape = shape;

            Output = input.Output;
            if (input.Derivatives != null)
            {
                Derivatives = input.Derivatives;
            }
            else
            {
                Derivatives = new Tensor(this, Output.GetShape());
            }

            oldShape = input.Output.GetShape();
            newShape = shape;

            Forward();
        }

        internal override void Forward()
        {
            Output.SetShape(newShape);
            Derivatives.SetShape(newShape);
        }

        internal override void Backward()
        {
            Output.SetShape(oldShape);
            Derivatives.SetShape(oldShape);
        }
    }
}
