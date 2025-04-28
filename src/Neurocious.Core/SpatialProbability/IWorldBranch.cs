using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.SpatialProbability
{
    public interface IWorldBranch
    {
        void UpdateValue(float newValue);
    }
}
