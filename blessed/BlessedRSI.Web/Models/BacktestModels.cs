using System.ComponentModel.DataAnnotations;

namespace BlessedRSI.Web.Models;

public class BacktestParameters
{
    [Range(1, 100)]
    public decimal BuyThreshold { get; set; } = 40.0m;
    
    [Range(50, 100)]
    public decimal SellTrigger { get; set; } = 90.0m;
    
    [Range(1, 12)]
    public int SellDelayWeeks { get; set; } = 4;
    
    [Range(0.01, 0.5)]
    public decimal PositionSizePct { get; set; } = 0.24m;
    
    [Range(0.05, 0.5)]
    public decimal StopLossPct { get; set; } = 0.24m;
    
    [Range(10000, 1000000)]
    public decimal StartingCapital { get; set; } = 100000m;
    
    public List<string> Symbols { get; set; } = new();
    
    public DateTime StartDate { get; set; } = DateTime.Now.AddYears(-2);
    
    public DateTime EndDate { get; set; } = DateTime.Now;
}

public class BacktestResult
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    [StringLength(100)]
    public string? StrategyName { get; set; }
    
    // Performance Metrics
    public decimal TotalReturn { get; set; }
    public decimal SortinoRatio { get; set; }
    public decimal SharpeRatio { get; set; }
    public decimal WinRate { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal AnnualizedReturn { get; set; }
    public decimal DownsideVolatility { get; set; }
    
    // Parameters used
    public decimal BuyThreshold { get; set; }
    public decimal SellTrigger { get; set; }
    public int SellDelayWeeks { get; set; }
    public decimal PositionSizePct { get; set; }
    public decimal StopLossPct { get; set; }
    public decimal StartingCapital { get; set; }
    
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    
    // JSON storage for detailed results
    public string? DetailedResultsJson { get; set; }
    
    public bool IsPublic { get; set; }
    
    [StringLength(500)]
    public string? Notes { get; set; }
}

public class StrategyShare
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    public int BacktestResultId { get; set; }
    public BacktestResult BacktestResult { get; set; } = null!;
    
    [StringLength(100)]
    public string StrategyName { get; set; } = string.Empty;
    
    [StringLength(100)]
    public string BiblicalName { get; set; } = string.Empty; // "David's Sling", "Solomon's Wisdom"
    
    [StringLength(1000)]
    public string Description { get; set; } = string.Empty;
    
    [StringLength(500)]
    public string? RelatedVerse { get; set; }
    
    public DateTime SharedAt { get; set; } = DateTime.UtcNow;
    
    public int Likes { get; set; }
    public int Views { get; set; }
    
    public ICollection<StrategyComment> Comments { get; set; } = new List<StrategyComment>();
}

public class StrategyComment
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    public int StrategyShareId { get; set; }
    public StrategyShare StrategyShare { get; set; } = null!;
    
    public string Content { get; set; } = string.Empty;
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}