using Microsoft.AspNetCore.Identity;
using System.ComponentModel.DataAnnotations;

namespace BlessedRSI.Web.Models;

public class ApplicationUser : IdentityUser
{
    [StringLength(100)]
    public string? FirstName { get; set; }
    
    [StringLength(100)]
    public string? LastName { get; set; }
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public SubscriptionTier SubscriptionTier { get; set; } = SubscriptionTier.Sparrow;
    
    public DateTime? SubscriptionExpiresAt { get; set; }
    
    [StringLength(500)]
    public string? Bio { get; set; }
    
    [StringLength(200)]
    public string? FavoriteVerse { get; set; }
    
    public int TotalBacktests { get; set; }
    
    public decimal BestSortinoRatio { get; set; }
    
    public int CommunityPoints { get; set; }
    
    // Two-Factor Authentication
    public bool TwoFactorEnabled { get; set; } = false;
    
    public DateTime? TwoFactorEnabledAt { get; set; }
    
    // Navigation properties
    public ICollection<BacktestResult> BacktestResults { get; set; } = new List<BacktestResult>();
    public ICollection<StrategyShare> SharedStrategies { get; set; } = new List<StrategyShare>();
    public ICollection<CommunityPost> CommunityPosts { get; set; } = new List<CommunityPost>();
    public TwoFactorSettings? TwoFactorSettings { get; set; }
    public ICollection<TwoFactorAuthenticationCode> TwoFactorCodes { get; set; } = new List<TwoFactorAuthenticationCode>();
    public ICollection<TwoFactorBackupCode> BackupCodes { get; set; } = new List<TwoFactorBackupCode>();
}

public enum SubscriptionTier
{
    Sparrow = 0,     // Free - "Consider the sparrows..." (Luke 12:24)
    Lion = 1,        // $29/month - "Bold as a lion" (Proverbs 28:1)
    Eagle = 2,       // $99/month - "Soar on wings like eagles" (Isaiah 40:31)
    Shepherd = 3     // $299/month - Ultimate leadership/guidance tier
}