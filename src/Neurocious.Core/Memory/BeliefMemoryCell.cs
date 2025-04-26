using Neurocious.Core.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neurocious.Core.Memory
{
    public class BeliefMemoryCell
    {
        public string BeliefId { get; set; }
        public float[] LatentVector { get; set; }
        public FieldParameters FieldParams { get; set; }
        public float Conviction { get; set; }
        public DateTime Created { get; set; }
        public DateTime LastAccessed { get; set; }
        public HashSet<string> NarrativeContexts { get; set; }
        public float DecayRate { get; set; }
        public Dictionary<string, float> MetaParameters { get; set; }
        public List<string> CausalAntecedents { get; set; }
        public List<string> CausalConsequences { get; set; }

        public float RetentionScore =>
            Conviction * (1.0f / (1.0f + DecayRate)) *
            Math.Max(0.1f, MetaParameters.GetValueOrDefault("narrative_coherence", 1.0f));

        public BeliefMemoryCell Clone() => new()
        {
            BeliefId = BeliefId,
            LatentVector = LatentVector.ToArray(),
            FieldParams = FieldParams,
            Conviction = Conviction,
            Created = Created,
            LastAccessed = LastAccessed,
            NarrativeContexts = new HashSet<string>(NarrativeContexts),
            DecayRate = DecayRate,
            MetaParameters = new Dictionary<string, float>(MetaParameters),
            CausalAntecedents = new List<string>(CausalAntecedents),
            CausalConsequences = new List<string>(CausalConsequences)
        };
    }
}
