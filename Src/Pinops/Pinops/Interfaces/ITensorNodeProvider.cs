using Pinops.Core.Nodes;

namespace Pinops.Core.Interfaces
{
    public interface ITensorNodeProvider
    {
        public Tensor GetTensor();
        public Node GetNode();
    }
}
