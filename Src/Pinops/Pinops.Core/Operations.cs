using Pinops.Core.Interfaces;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;

namespace Pinops.Core
{
    public static class Operations
    {
        public static Operation OneHot(this ITensorNodeProvider indices, Graph graph = null)
        {
            return new OneHot(graph, indices.GetNode());
        }

        public static Operation Argmax(this ITensorNodeProvider input, int axis = -1, Graph graph = null)
        {
            return new Argmax(graph, input.GetNode(), axis);
        }
    }
}
