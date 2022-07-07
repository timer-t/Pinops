using System;
using System.Collections.Generic;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core.Optimizers
{
    public abstract class Optimizer
    {
        internal abstract void Init(List<Variable> variables);
        internal abstract void Minimize(List<Variable> variables);
    }
}
