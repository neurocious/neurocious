namespace Neurocious.Core.SpatialProbabilityNetwork;

public class FlowPattern
{
    public float[] Position { get; set; }
    public float[] FlowDirection { get; set; }
    public float LocalCurvature { get; set; }
    public float LocalEntropy { get; set; }
    public float LocalAlignment { get; set; }
    public float Stability { get; set; }
}
