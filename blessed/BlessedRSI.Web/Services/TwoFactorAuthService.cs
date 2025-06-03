using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using System.Security.Cryptography;
using System.Text;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class TwoFactorAuthService
{
    private readonly ApplicationDbContext _context;
    private readonly UserManager<ApplicationUser> _userManager;
    private readonly EmailService _emailService;
    private readonly ILogger<TwoFactorAuthService> _logger;
    private readonly IHttpContextAccessor _httpContextAccessor;

    public TwoFactorAuthService(
        ApplicationDbContext context,
        UserManager<ApplicationUser> userManager,
        EmailService emailService,
        ILogger<TwoFactorAuthService> logger,
        IHttpContextAccessor httpContextAccessor)
    {
        _context = context;
        _userManager = userManager;
        _emailService = emailService;
        _logger = logger;
        _httpContextAccessor = httpContextAccessor;
    }

    public async Task<string> GenerateAndSendTwoFactorCodeAsync(ApplicationUser user, TwoFactorCodeType codeType)
    {
        // Invalidate any existing codes of the same type
        await InvalidateExistingCodesAsync(user.Id, codeType);

        // Generate new 6-digit code
        var code = GenerateNumericCode(6);
        var expirationMinutes = GetCodeExpirationMinutes(codeType);

        // Store in database
        var twoFactorCode = new TwoFactorAuthenticationCode
        {
            UserId = user.Id,
            Code = code,
            CodeType = codeType,
            ExpiresAt = DateTime.UtcNow.AddMinutes(expirationMinutes),
            IpAddress = GetClientIpAddress(),
            UserAgent = GetUserAgent()
        };

        _context.TwoFactorAuthenticationCodes.Add(twoFactorCode);
        await _context.SaveChangesAsync();

        // Send email
        var userName = $"{user.FirstName} {user.LastName}".Trim();
        var emailSent = await _emailService.SendTwoFactorCodeAsync(
            user.Email!, userName, code, codeType, GetClientIpAddress());

        if (!emailSent)
        {
            _logger.LogError("Failed to send 2FA code email to user {UserId}", user.Id);
            throw new InvalidOperationException("Failed to send verification code");
        }

        _logger.LogInformation("2FA code generated and sent for user {UserId}, type {CodeType}", user.Id, codeType);
        return code;
    }

    public async Task<bool> VerifyTwoFactorCodeAsync(string userId, string code, TwoFactorCodeType codeType)
    {
        var storedCode = await _context.TwoFactorAuthenticationCodes
            .Where(c => c.UserId == userId && 
                       c.Code == code && 
                       c.CodeType == codeType && 
                       c.UsedAt == null && 
                       c.ExpiresAt > DateTime.UtcNow)
            .FirstOrDefaultAsync();

        if (storedCode == null)
        {
            _logger.LogWarning("Invalid 2FA code attempt for user {UserId}, type {CodeType}", userId, codeType);
            return false;
        }

        // Mark code as used
        storedCode.UsedAt = DateTime.UtcNow;
        await _context.SaveChangesAsync();

        _logger.LogInformation("2FA code verified successfully for user {UserId}, type {CodeType}", userId, codeType);
        return true;
    }

    public async Task<TwoFactorResponse> EnableTwoFactorAsync(string userId, EnableTwoFactorRequest request)
    {
        try
        {
            var user = await _userManager.FindByIdAsync(userId);
            if (user == null)
            {
                return new TwoFactorResponse
                {
                    Success = false,
                    Message = "User not found"
                };
            }

            // Verify the email code first
            var codeValid = await VerifyTwoFactorCodeAsync(userId, request.VerificationCode, TwoFactorCodeType.EmailVerification);
            if (!codeValid)
            {
                return new TwoFactorResponse
                {
                    Success = false,
                    Message = "Invalid or expired verification code"
                };
            }

            // Create or update 2FA settings
            var settings = await _context.TwoFactorSettings.FirstOrDefaultAsync(s => s.UserId == userId);
            if (settings == null)
            {
                settings = new TwoFactorSettings
                {
                    UserId = userId,
                    IsEmailTwoFactorEnabled = true,
                    RequireTwoFactorForLogin = request.RequireForLogin,
                    RequireTwoFactorForSensitiveActions = request.RequireForSensitiveActions,
                    EnabledAt = DateTime.UtcNow
                };
                _context.TwoFactorSettings.Add(settings);
            }
            else
            {
                settings.IsEmailTwoFactorEnabled = true;
                settings.RequireTwoFactorForLogin = request.RequireForLogin;
                settings.RequireTwoFactorForSensitiveActions = request.RequireForSensitiveActions;
                settings.EnabledAt = DateTime.UtcNow;
                settings.UpdatedAt = DateTime.UtcNow;
            }

            // Update user
            user.TwoFactorEnabled = true;
            user.TwoFactorEnabledAt = DateTime.UtcNow;

            // Generate backup codes
            var backupCodes = await GenerateBackupCodesAsync(userId);
            
            await _context.SaveChangesAsync();

            // Send notification email
            var userName = $"{user.FirstName} {user.LastName}".Trim();
            await _emailService.SendTwoFactorEnabledNotificationAsync(user.Email!, userName);

            _logger.LogInformation("Two-factor authentication enabled for user {UserId}", userId);

            return new TwoFactorResponse
            {
                Success = true,
                Message = "Two-factor authentication has been enabled successfully"
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enabling 2FA for user {UserId}", userId);
            return new TwoFactorResponse
            {
                Success = false,
                Message = "An error occurred while enabling two-factor authentication"
            };
        }
    }

    public async Task<TwoFactorResponse> DisableTwoFactorAsync(string userId, DisableTwoFactorRequest request)
    {
        try
        {
            var user = await _userManager.FindByIdAsync(userId);
            if (user == null)
            {
                return new TwoFactorResponse
                {
                    Success = false,
                    Message = "User not found"
                };
            }

            // Verify password
            var passwordValid = await _userManager.CheckPasswordAsync(user, request.Password);
            if (!passwordValid)
            {
                return new TwoFactorResponse
                {
                    Success = false,
                    Message = "Invalid password"
                };
            }

            // Verify either 2FA code or backup code
            bool authenticationVerified = false;

            if (!string.IsNullOrEmpty(request.VerificationCode))
            {
                authenticationVerified = await VerifyTwoFactorCodeAsync(userId, request.VerificationCode, TwoFactorCodeType.SecurityAction);
            }
            else if (!string.IsNullOrEmpty(request.BackupCode))
            {
                authenticationVerified = await UseBackupCodeAsync(userId, request.BackupCode);
            }

            if (!authenticationVerified)
            {
                return new TwoFactorResponse
                {
                    Success = false,
                    Message = "Invalid verification code or backup code"
                };
            }

            // Disable 2FA
            var settings = await _context.TwoFactorSettings.FirstOrDefaultAsync(s => s.UserId == userId);
            if (settings != null)
            {
                settings.IsEmailTwoFactorEnabled = false;
                settings.UpdatedAt = DateTime.UtcNow;
            }

            user.TwoFactorEnabled = false;

            // Invalidate all backup codes
            var backupCodes = await _context.TwoFactorBackupCodes
                .Where(bc => bc.UserId == userId && !bc.IsUsed)
                .ToListAsync();

            foreach (var code in backupCodes)
            {
                code.UsedAt = DateTime.UtcNow;
                code.UsedFromIp = GetClientIpAddress();
            }

            await _context.SaveChangesAsync();

            // Send notification email
            var userName = $"{user.FirstName} {user.LastName}".Trim();
            await _emailService.SendTwoFactorDisabledNotificationAsync(user.Email!, userName, GetClientIpAddress());

            _logger.LogInformation("Two-factor authentication disabled for user {UserId}", userId);

            return new TwoFactorResponse
            {
                Success = true,
                Message = "Two-factor authentication has been disabled"
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disabling 2FA for user {UserId}", userId);
            return new TwoFactorResponse
            {
                Success = false,
                Message = "An error occurred while disabling two-factor authentication"
            };
        }
    }

    public async Task<BackupCodesResponse> GenerateNewBackupCodesAsync(string userId, GenerateBackupCodesRequest request)
    {
        try
        {
            var user = await _userManager.FindByIdAsync(userId);
            if (user == null)
            {
                return new BackupCodesResponse
                {
                    Success = false,
                    Message = "User not found"
                };
            }

            // Verify password
            var passwordValid = await _userManager.CheckPasswordAsync(user, request.Password);
            if (!passwordValid)
            {
                return new BackupCodesResponse
                {
                    Success = false,
                    Message = "Invalid password"
                };
            }

            // Verify 2FA code if provided
            if (!string.IsNullOrEmpty(request.VerificationCode))
            {
                var codeValid = await VerifyTwoFactorCodeAsync(userId, request.VerificationCode, TwoFactorCodeType.SecurityAction);
                if (!codeValid)
                {
                    return new BackupCodesResponse
                    {
                        Success = false,
                        Message = "Invalid verification code"
                    };
                }
            }

            // Generate new backup codes
            var newCodes = await GenerateBackupCodesAsync(userId, invalidateExisting: true);
            
            // Send backup codes via email
            var userName = $"{user.FirstName} {user.LastName}".Trim();
            await _emailService.SendBackupCodesAsync(user.Email!, userName, newCodes);

            return new BackupCodesResponse
            {
                Success = true,
                Message = "New backup codes generated successfully",
                BackupCodes = newCodes
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating backup codes for user {UserId}", userId);
            return new BackupCodesResponse
            {
                Success = false,
                Message = "An error occurred while generating backup codes"
            };
        }
    }

    public async Task<bool> UseBackupCodeAsync(string userId, string backupCode)
    {
        var code = await _context.TwoFactorBackupCodes
            .Where(bc => bc.UserId == userId && 
                        bc.Code == backupCode && 
                        bc.UsedAt == null)
            .FirstOrDefaultAsync();

        if (code == null)
        {
            _logger.LogWarning("Invalid backup code attempt for user {UserId}", userId);
            return false;
        }

        // Mark code as used
        code.UsedAt = DateTime.UtcNow;
        code.UsedFromIp = GetClientIpAddress();

        // Update remaining backup codes count
        var settings = await _context.TwoFactorSettings.FirstOrDefaultAsync(s => s.UserId == userId);
        if (settings != null)
        {
            settings.BackupCodesRemaining = Math.Max(0, settings.BackupCodesRemaining - 1);
        }

        await _context.SaveChangesAsync();

        _logger.LogInformation("Backup code used successfully for user {UserId}", userId);
        return true;
    }

    public async Task<TwoFactorStatusResponse> GetTwoFactorStatusAsync(string userId)
    {
        var user = await _context.Users
            .Include(u => u.TwoFactorSettings)
            .FirstOrDefaultAsync(u => u.Id == userId);

        if (user == null)
        {
            return new TwoFactorStatusResponse();
        }

        var unusedBackupCodes = await _context.TwoFactorBackupCodes
            .CountAsync(bc => bc.UserId == userId && bc.UsedAt == null);

        return new TwoFactorStatusResponse
        {
            IsEnabled = user.TwoFactorEnabled,
            RequiredForLogin = user.TwoFactorSettings?.RequireTwoFactorForLogin ?? false,
            RequiredForSensitiveActions = user.TwoFactorSettings?.RequireTwoFactorForSensitiveActions ?? false,
            EnabledAt = user.TwoFactorEnabledAt,
            BackupCodesRemaining = unusedBackupCodes,
            LastBackupCodesGenerated = user.TwoFactorSettings?.LastBackupCodesGeneratedAt,
            HasUnusedBackupCodes = unusedBackupCodes > 0
        };
    }

    private async Task<List<string>> GenerateBackupCodesAsync(string userId, bool invalidateExisting = true)
    {
        if (invalidateExisting)
        {
            // Mark existing backup codes as used
            var existingCodes = await _context.TwoFactorBackupCodes
                .Where(bc => bc.UserId == userId && bc.UsedAt == null)
                .ToListAsync();

            foreach (var code in existingCodes)
            {
                code.UsedAt = DateTime.UtcNow;
                code.UsedFromIp = GetClientIpAddress();
            }
        }

        // Generate 10 new backup codes
        var newCodes = new List<string>();
        for (int i = 0; i < 10; i++)
        {
            var code = GenerateAlphanumericCode(8);
            newCodes.Add(code);

            _context.TwoFactorBackupCodes.Add(new TwoFactorBackupCode
            {
                UserId = userId,
                Code = code
            });
        }

        // Update settings
        var settings = await _context.TwoFactorSettings.FirstOrDefaultAsync(s => s.UserId == userId);
        if (settings != null)
        {
            settings.LastBackupCodesGeneratedAt = DateTime.UtcNow;
            settings.BackupCodesRemaining = 10;
            settings.UpdatedAt = DateTime.UtcNow;
        }

        await _context.SaveChangesAsync();

        _logger.LogInformation("Generated {Count} backup codes for user {UserId}", newCodes.Count, userId);
        return newCodes;
    }

    private async Task InvalidateExistingCodesAsync(string userId, TwoFactorCodeType codeType)
    {
        var existingCodes = await _context.TwoFactorAuthenticationCodes
            .Where(c => c.UserId == userId && c.CodeType == codeType && c.UsedAt == null)
            .ToListAsync();

        foreach (var code in existingCodes)
        {
            code.UsedAt = DateTime.UtcNow;
        }

        if (existingCodes.Any())
        {
            await _context.SaveChangesAsync();
        }
    }

    private static string GenerateNumericCode(int length)
    {
        using var rng = RandomNumberGenerator.Create();
        var bytes = new byte[4];
        rng.GetBytes(bytes);
        var number = BitConverter.ToUInt32(bytes, 0);
        var code = (number % (uint)Math.Pow(10, length)).ToString().PadLeft(length, '0');
        return code;
    }

    private static string GenerateAlphanumericCode(int length)
    {
        const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        using var rng = RandomNumberGenerator.Create();
        var stringBuilder = new StringBuilder();
        var bytes = new byte[length];
        rng.GetBytes(bytes);

        foreach (var b in bytes)
        {
            stringBuilder.Append(chars[b % chars.Length]);
        }

        return stringBuilder.ToString();
    }

    private static int GetCodeExpirationMinutes(TwoFactorCodeType codeType) => codeType switch
    {
        TwoFactorCodeType.Login => 10,
        TwoFactorCodeType.EmailVerification => 30,
        TwoFactorCodeType.PasswordReset => 15,
        TwoFactorCodeType.EmailChange => 30,
        TwoFactorCodeType.SecurityAction => 15,
        _ => 15
    };

    private string? GetClientIpAddress()
    {
        var context = _httpContextAccessor.HttpContext;
        if (context == null) return null;

        var ipAddress = context.Request.Headers["X-Forwarded-For"].FirstOrDefault();
        if (string.IsNullOrEmpty(ipAddress))
        {
            ipAddress = context.Request.Headers["X-Real-IP"].FirstOrDefault();
        }
        if (string.IsNullOrEmpty(ipAddress))
        {
            ipAddress = context.Connection.RemoteIpAddress?.ToString();
        }

        return ipAddress;
    }

    private string? GetUserAgent()
    {
        var context = _httpContextAccessor.HttpContext;
        return context?.Request.Headers["User-Agent"].FirstOrDefault();
    }
}