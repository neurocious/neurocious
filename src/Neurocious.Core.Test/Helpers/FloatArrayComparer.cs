using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Test.Helpers
{
    public class FloatArrayComparer : IEqualityComparer<double[]>
    {
        private readonly float epsilon;

        public FloatArrayComparer(float epsilon)
        {
            this.epsilon = epsilon;
        }

        public bool Equals(double[] x, double[] y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (x == null || y == null) return false;
            if (x.Length != y.Length) return false;

            return !x.Where((t, i) => Math.Abs(t - y[i]) > epsilon).Any();
        }

        public int GetHashCode(double[] obj)
        {
            if (obj == null) return 0;
            return obj.Aggregate(17, (current, item) => current * 23 + item.GetHashCode());
        }
    }
}
