using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BlessedRSI.Web.Models;

public class UserApiKey
{
    [Key]
    public int Id { get; set; }
    
    [Required]
    [StringLength(128)]
    public string ApiKeyHash { get; set; } = string.Empty;
    
    [Required]
    [StringLength(50)]
    public string Name { get; set; } = string.Empty;
    
    [StringLength(255)]
    public string? Description { get; set; }
    
    [Required]
    public string UserId { get; set; } = string.Empty;
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public DateTime? LastUsedAt { get; set; }
    
    public DateTime? ExpiresAt { get; set; }
    
    public bool IsActive { get; set; } = true;
    
    public long RequestCount { get; set; } = 0;
    
    public long RequestLimitPerHour { get; set; } = 1000;
    
    [StringLength(50)]
    public string Prefix { get; set; } = string.Empty;
    
    // Navigation properties
    [ForeignKey(nameof(UserId))]
    public ApplicationUser User { get; set; } = null!;
    
    public ICollection<ApiKeyUsageLog> UsageLogs { get; set; } = new List<ApiKeyUsageLog>();
}

public class ApiKeyUsageLog
{
    [Key]
    public int Id { get; set; }
    
    [Required]
    public int ApiKeyId { get; set; }
    
    [Required]
    [StringLength(100)]
    public string Endpoint { get; set; } = string.Empty;
    
    [StringLength(10)]
    public string Method { get; set; } = string.Empty;
    
    public int StatusCode { get; set; }
    
    [StringLength(45)]
    public string? IpAddress { get; set; }
    
    [StringLength(500)]
    public string? UserAgent { get; set; }
    
    public DateTime RequestedAt { get; set; } = DateTime.UtcNow;
    
    public int ResponseTimeMs { get; set; }
    
    // Navigation properties
    [ForeignKey(nameof(ApiKeyId))]
    public UserApiKey ApiKey { get; set; } = null!;
}

// Request/Response models for API
public class CreateApiKeyRequest
{
    [Required]
    [StringLength(50, MinimumLength = 3)]
    public string Name { get; set; } = string.Empty;
    
    [StringLength(255)]
    public string? Description { get; set; }
    
    public DateTime? ExpiresAt { get; set; }
    
    public long? RequestLimitPerHour { get; set; }
}

public class CreateApiKeyResponse
{
    public bool Success { get; set; }
    public string? Message { get; set; }
    public string? ApiKey { get; set; }
    public UserApiKeyDto? KeyInfo { get; set; }
    public List<string> Errors { get; set; } = new();
}

public class UserApiKeyDto
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string? Description { get; set; }
    public string Prefix { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public DateTime? LastUsedAt { get; set; }
    public DateTime? ExpiresAt { get; set; }
    public bool IsActive { get; set; }
    public long RequestCount { get; set; }
    public long RequestLimitPerHour { get; set; }
    public bool IsExpired => ExpiresAt.HasValue && ExpiresAt.Value < DateTime.UtcNow;
}

public class UpdateApiKeyRequest
{
    [StringLength(50, MinimumLength = 3)]
    public string? Name { get; set; }
    
    [StringLength(255)]
    public string? Description { get; set; }
    
    public bool? IsActive { get; set; }
    
    public DateTime? ExpiresAt { get; set; }
    
    public long? RequestLimitPerHour { get; set; }
}

public class ApiKeyListResponse
{
    public bool Success { get; set; }
    public List<UserApiKeyDto> ApiKeys { get; set; } = new();
    public string? Message { get; set; }
}