using System.ComponentModel.DataAnnotations;

namespace BlessedRSI.Web.Models;

public class CommunityPost
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    [StringLength(200)]
    public string Title { get; set; } = string.Empty;
    
    public string Content { get; set; } = string.Empty;
    
    public PostType Type { get; set; }
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public int Likes { get; set; }
    public int Views { get; set; }
    
    public bool IsPinned { get; set; }
    
    [StringLength(500)]
    public string? RelatedVerse { get; set; }
    
    public ICollection<PostComment> Comments { get; set; } = new List<PostComment>();
    public ICollection<PostLike> PostLikes { get; set; } = new List<PostLike>();
}

public enum PostType
{
    Discussion,
    PrayerRequest,
    Testimony,
    Question,
    MarketInsight,
    BiblicalReflection
}

public class PostComment
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    public int PostId { get; set; }
    public CommunityPost Post { get; set; } = null!;
    
    public string Content { get; set; } = string.Empty;
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public int Likes { get; set; }
}

public class PostLike
{
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    public int PostId { get; set; }
    public CommunityPost Post { get; set; } = null!;
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

public class Achievement
{
    public int Id { get; set; }
    
    [StringLength(100)]
    public string Name { get; set; } = string.Empty;
    
    [StringLength(500)]
    public string Description { get; set; } = string.Empty;
    
    [StringLength(100)]
    public string IconClass { get; set; } = string.Empty;
    
    public AchievementType Type { get; set; }
    
    public decimal RequiredValue { get; set; }
    
    public int Points { get; set; }
}

public enum AchievementType
{
    FirstBacktest,
    SortinoRatio,
    WinRate,
    TotalReturn,
    CommunityEngagement,
    Consistency
}

public class UserAchievement
{
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    public int AchievementId { get; set; }
    public Achievement Achievement { get; set; } = null!;
    
    public DateTime EarnedAt { get; set; } = DateTime.UtcNow;
}

// Additional models for leaderboard
public class LeaderboardEntry
{
    public string UserName { get; set; } = string.Empty;
    public string? FavoriteVerse { get; set; }
    public string? BestStrategyName { get; set; }
    public decimal BestSortinoRatio { get; set; }
    public decimal BestTotalReturn { get; set; }
    public decimal BestWinRate { get; set; }
    public int CommunityPoints { get; set; }
    public DateTime LastActive { get; set; }
    public List<Achievement> TopAchievements { get; set; } = new();
}

public class FeaturedStrategy
{
    public int Id { get; set; }
    public string BiblicalName { get; set; } = string.Empty;
    public string CreatedBy { get; set; } = string.Empty;
    public decimal SortinoRatio { get; set; }
    public string? RelatedVerse { get; set; }
}

public class RecentAchievement
{
    public string UserName { get; set; } = string.Empty;
    public string AchievementName { get; set; } = string.Empty;
    public string IconClass { get; set; } = string.Empty;
}

public class CommunityStats
{
    public int TotalMembers { get; set; }
    public int TotalBacktests { get; set; }
    public decimal AvgSortinoRatio { get; set; }
    public int SharedStrategies { get; set; }
}

public class ActiveMember
{
    public string Name { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public bool IsOnline { get; set; }
}