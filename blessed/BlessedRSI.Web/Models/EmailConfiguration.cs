using System.ComponentModel.DataAnnotations;

namespace BlessedRSI.Web.Models;

public class EmailConfiguration
{
    [Required]
    public string SmtpHost { get; set; } = string.Empty;
    
    [Range(1, 65535)]
    public int SmtpPort { get; set; } = 587;
    
    public string? SmtpUsername { get; set; }
    
    public string? SmtpPassword { get; set; }
    
    [Required]
    [EmailAddress]
    public string FromEmail { get; set; } = string.Empty;
    
    [Required]
    public string FromName { get; set; } = string.Empty;
    
    [EmailAddress]
    public string? SupportEmail { get; set; }
    
    public bool EnableSsl { get; set; } = true;
    
    public bool RequireAuthentication { get; set; } = true;
    
    public int TimeoutSeconds { get; set; } = 30;
    
    public int MaxRetries { get; set; } = 3;
    
    public int RetryDelaySeconds { get; set; } = 5;
    
    // Fallback email settings
    public EmailConfiguration? FallbackConfiguration { get; set; }
    
    public bool UseFallbackOnFailure { get; set; } = true;
}

public class EmailValidationResult
{
    public bool IsValid { get; set; }
    public bool CanConnect { get; set; }
    public bool CanAuthenticate { get; set; }
    public bool CanSendTest { get; set; }
    public List<string> Errors { get; set; } = new();
    public List<string> Warnings { get; set; } = new();
    public TimeSpan TestDuration { get; set; }
    public string? TestEmailMessageId { get; set; }
}

public class EmailSendResult
{
    public bool Success { get; set; }
    public string? MessageId { get; set; }
    public List<string> Errors { get; set; } = new();
    public bool UsedFallback { get; set; }
    public string? Provider { get; set; }
    public TimeSpan SendDuration { get; set; }
    public int AttemptCount { get; set; }
}

public class EmailHealthStatus
{
    public bool IsHealthy { get; set; }
    public string Provider { get; set; } = string.Empty;
    public DateTime LastSuccessful { get; set; }
    public DateTime LastAttempt { get; set; }
    public int ConsecutiveFailures { get; set; }
    public TimeSpan AverageResponseTime { get; set; }
    public string? LastError { get; set; }
    public bool FallbackAvailable { get; set; }
}

public enum EmailProvider
{
    Primary,
    Fallback,
    Local // For development/testing
}

public enum EmailTemplate
{
    TwoFactorCode,
    BackupCodes,
    SecurityAlert,
    WelcomeEmail,
    PasswordReset,
    AccountVerification
}