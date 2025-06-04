using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BlessedRSI.Web.Models;

public class RateLimitEntry
{
    [Key]
    public int Id { get; set; }
    
    [Required]
    [StringLength(45)]
    public string IpAddress { get; set; } = string.Empty;
    
    [StringLength(100)]
    public string? UserId { get; set; }
    
    [Required]
    [StringLength(200)]
    public string Endpoint { get; set; } = string.Empty;
    
    public DateTime WindowStart { get; set; }
    
    public int RequestCount { get; set; }
    
    public DateTime LastRequest { get; set; }
    
    [StringLength(50)]
    public string RateLimitType { get; set; } = "General";
    
    // Navigation property
    [ForeignKey(nameof(UserId))]
    public ApplicationUser? User { get; set; }
}

public class RateLimitConfig
{
    public string Name { get; set; } = string.Empty;
    public int RequestLimit { get; set; }
    public TimeSpan WindowSize { get; set; }
    public string[] Endpoints { get; set; } = Array.Empty<string>();
    public SubscriptionTier? RequiredTier { get; set; }
    public bool ApplyToAuthenticated { get; set; } = true;
    public bool ApplyToAnonymous { get; set; } = true;
}

public class RateLimitResult
{
    public bool IsAllowed { get; set; }
    public int RequestsRemaining { get; set; }
    public int RequestLimit { get; set; }
    public TimeSpan WindowSize { get; set; }
    public DateTime WindowReset { get; set; }
    public string Message { get; set; } = string.Empty;
}

public class RateLimitSettings
{
    public bool Enabled { get; set; } = true;
    public bool UseRedis { get; set; } = false;
    public string? RedisConnectionString { get; set; }
    public int DefaultWindowSizeMinutes { get; set; } = 60;
    public int CleanupIntervalMinutes { get; set; } = 30;
    
    // Default limits by subscription tier
    public int SparrowRequestsPerHour { get; set; } = 100;
    public int LionRequestsPerHour { get; set; } = 1000;
    public int EagleRequestsPerHour { get; set; } = 5000;
    public int ShepherdRequestsPerHour { get; set; } = 10000;
    
    // Anonymous user limits
    public int AnonymousRequestsPerHour { get; set; } = 50;
    public int AnonymousRequestsPerMinute { get; set; } = 10;
    
    // Specific endpoint limits
    public int AuthEndpointRequestsPerMinute { get; set; } = 5;
    public int RegistrationRequestsPerHour { get; set; } = 3;
    public int PasswordResetRequestsPerHour { get; set; } = 2;
}