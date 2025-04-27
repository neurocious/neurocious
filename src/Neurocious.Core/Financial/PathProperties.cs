using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Financial
{
    public class PathProperties
    {
        public double Smoothness { get; set; }
        public double Consistency { get; set; }
        public double Efficiency { get; set; }
        public List<double[]> Velocities { get; set; }
        public List<double[]> Accelerations { get; set; }
        public double TotalLength { get; set; }
        public double DirectDistance { get; set; }
        public Dictionary<string, double> AdditionalMetrics { get; set; }

        public PathProperties()
        {
            Velocities = new List<double[]>();
            Accelerations = new List<double[]>();
            AdditionalMetrics = new Dictionary<string, double>();
        }

        public double CalculateOverallQuality()
        {
            return (Smoothness + Consistency + Efficiency) / 3.0;
        }
    }
}
