using ParallelReverseAutoDiff.PRAD;

namespace Neurocious.Core.Common
{
    public static class PradOpExtensions
    {
        public static PradResult Then(this PradOp pradOp, Func<PradOp, PradResult> func)
        {
            return func.Invoke(pradOp);
        }
    }
}
