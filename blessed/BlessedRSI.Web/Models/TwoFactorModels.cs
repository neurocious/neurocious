using System.ComponentModel.DataAnnotations;

namespace BlessedRSI.Web.Models;

public class TwoFactorAuthenticationCode
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    [StringLength(8)]
    public string Code { get; set; } = string.Empty;
    
    public TwoFactorCodeType CodeType { get; set; }
    
    public DateTime ExpiresAt { get; set; }
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public DateTime? UsedAt { get; set; }
    
    public string? IpAddress { get; set; }
    
    public string? UserAgent { get; set; }
    
    public bool IsExpired => DateTime.UtcNow > ExpiresAt;
    
    public bool IsUsed => UsedAt.HasValue;
    
    public bool IsValid => !IsExpired && !IsUsed;
}

public enum TwoFactorCodeType
{
    EmailVerification,
    Login,
    PasswordReset,
    EmailChange,
    SecurityAction
}

public class TwoFactorBackupCode
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    [StringLength(12)]
    public string Code { get; set; } = string.Empty;
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    public DateTime? UsedAt { get; set; }
    
    public string? UsedFromIp { get; set; }
    
    public bool IsUsed => UsedAt.HasValue;
}

public class TwoFactorSettings
{
    public int Id { get; set; }
    
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;
    
    public bool IsEmailTwoFactorEnabled { get; set; }
    
    public bool RequireTwoFactorForLogin { get; set; }
    
    public bool RequireTwoFactorForSensitiveActions { get; set; }
    
    public DateTime? EnabledAt { get; set; }
    
    public DateTime? LastBackupCodesGeneratedAt { get; set; }
    
    public int BackupCodesRemaining { get; set; }
    
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}

// Request/Response Models
public class EnableTwoFactorRequest
{
    [Required(ErrorMessage = "Email verification code is required")]
    [StringLength(8, MinimumLength = 6, ErrorMessage = "Verification code must be 6-8 characters")]
    public string VerificationCode { get; set; } = string.Empty;
    
    public bool RequireForLogin { get; set; } = true;
    
    public bool RequireForSensitiveActions { get; set; } = true;
}

public class DisableTwoFactorRequest
{
    [Required(ErrorMessage = "Password is required to disable 2FA")]
    [DataType(DataType.Password)]
    public string Password { get; set; } = string.Empty;
    
    [StringLength(8, MinimumLength = 6, ErrorMessage = "Verification code must be 6-8 characters")]
    public string? VerificationCode { get; set; }
    
    [StringLength(12, MinimumLength = 8, ErrorMessage = "Backup code must be 8-12 characters")]
    public string? BackupCode { get; set; }
}

public class VerifyTwoFactorRequest
{
    [Required(ErrorMessage = "Verification code is required")]
    [StringLength(8, MinimumLength = 6, ErrorMessage = "Verification code must be 6-8 characters")]
    public string Code { get; set; } = string.Empty;
    
    public bool RememberDevice { get; set; } = false;
    
    public string? ReturnUrl { get; set; }
}

public class UseTwoFactorBackupCodeRequest
{
    [Required(ErrorMessage = "Backup code is required")]
    [StringLength(12, MinimumLength = 8, ErrorMessage = "Backup code must be 8-12 characters")]
    public string BackupCode { get; set; } = string.Empty;
    
    public string? ReturnUrl { get; set; }
}

public class GenerateBackupCodesRequest
{
    [Required(ErrorMessage = "Password is required")]
    [DataType(DataType.Password)]
    public string Password { get; set; } = string.Empty;
    
    [StringLength(8, MinimumLength = 6, ErrorMessage = "Verification code must be 6-8 characters")]
    public string? VerificationCode { get; set; }
}

public class TwoFactorStatusResponse
{
    public bool IsEnabled { get; set; }
    public bool RequiredForLogin { get; set; }
    public bool RequiredForSensitiveActions { get; set; }
    public DateTime? EnabledAt { get; set; }
    public int BackupCodesRemaining { get; set; }
    public DateTime? LastBackupCodesGenerated { get; set; }
    public bool HasUnusedBackupCodes { get; set; }
}

public class BackupCodesResponse
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public List<string> BackupCodes { get; set; } = new();
    public List<string> Errors { get; set; } = new();
}

public class TwoFactorResponse
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public bool RequiresTwoFactor { get; set; }
    public bool BackupCodeUsed { get; set; }
    public int BackupCodesRemaining { get; set; }
    public List<string> Errors { get; set; } = new();
}

// Email template models
public class EmailVerificationModel
{
    public string UserName { get; set; } = string.Empty;
    public string VerificationCode { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public DateTime ExpiresAt { get; set; }
    public string SupportEmail { get; set; } = string.Empty;
}

public class TwoFactorCodeEmailModel
{
    public string UserName { get; set; } = string.Empty;
    public string VerificationCode { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public DateTime ExpiresAt { get; set; }
    public string IpAddress { get; set; } = string.Empty;
    public string Location { get; set; } = string.Empty;
    public string SupportEmail { get; set; } = string.Empty;
}

public class BackupCodesEmailModel
{
    public string UserName { get; set; } = string.Empty;
    public List<string> BackupCodes { get; set; } = new();
    public DateTime GeneratedAt { get; set; }
    public string SupportEmail { get; set; } = string.Empty;
}