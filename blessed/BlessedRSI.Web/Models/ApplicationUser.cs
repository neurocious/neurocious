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
    
    public SubscriptionTier SubscriptionTier { get; set; } = SubscriptionTier.Seeker;
    
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
    Seeker = 0,      // Free
    Believer = 1,    // $29/month
    Disciple = 2,    // $99/month
    Apostle = 3      // $299/month
}