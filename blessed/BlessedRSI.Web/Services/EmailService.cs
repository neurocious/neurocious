using FluentEmail.Core;
using FluentEmail.Core.Models;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class EmailService
{
    private readonly IFluentEmail _fluentEmail;
    private readonly IConfiguration _configuration;
    private readonly ILogger<EmailService> _logger;

    public EmailService(IFluentEmail fluentEmail, IConfiguration configuration, ILogger<EmailService> logger)
    {
        _fluentEmail = fluentEmail;
        _configuration = configuration;
        _logger = logger;
    }

    public async Task<bool> SendTwoFactorCodeAsync(string email, string userName, string code, TwoFactorCodeType codeType, string? ipAddress = null)
    {
        try
        {
            var emailSettings = _configuration.GetSection("EmailSettings");
            var action = GetActionDescription(codeType);
            
            var model = new TwoFactorCodeEmailModel
            {
                UserName = userName,
                VerificationCode = code,
                Action = action,
                ExpiresAt = DateTime.UtcNow.AddMinutes(GetCodeExpirationMinutes(codeType)),
                IpAddress = ipAddress ?? "Unknown",
                Location = "Unknown", // Could integrate with geolocation service
                SupportEmail = emailSettings["SupportEmail"] ?? "support@blessedrsi.com"
            };

            var result = await _fluentEmail
                .To(email)
                .Subject($"BlessedRSI - Verification Code for {action}")
                .UsingTemplate(GetTwoFactorCodeTemplate(), model)
                .SendAsync();

            if (result.Successful)
            {
                _logger.LogInformation("2FA code email sent successfully to {Email} for action {Action}", email, action);
                return true;
            }
            else
            {
                _logger.LogError("Failed to send 2FA code email to {Email}: {Errors}", 
                    email, string.Join(", ", result.ErrorMessages));
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending 2FA code email to {Email}", email);
            return false;
        }
    }

    public async Task<bool> SendBackupCodesAsync(string email, string userName, List<string> backupCodes)
    {
        try
        {
            var emailSettings = _configuration.GetSection("EmailSettings");
            
            var model = new BackupCodesEmailModel
            {
                UserName = userName,
                BackupCodes = backupCodes,
                GeneratedAt = DateTime.UtcNow,
                SupportEmail = emailSettings["SupportEmail"] ?? "support@blessedrsi.com"
            };

            var result = await _fluentEmail
                .To(email)
                .Subject("BlessedRSI - Your New Backup Codes")
                .UsingTemplate(GetBackupCodesTemplate(), model)
                .SendAsync();

            if (result.Successful)
            {
                _logger.LogInformation("Backup codes email sent successfully to {Email}", email);
                return true;
            }
            else
            {
                _logger.LogError("Failed to send backup codes email to {Email}: {Errors}", 
                    email, string.Join(", ", result.ErrorMessages));
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending backup codes email to {Email}", email);
            return false;
        }
    }

    public async Task<bool> SendTwoFactorEnabledNotificationAsync(string email, string userName)
    {
        try
        {
            var emailSettings = _configuration.GetSection("EmailSettings");
            
            var model = new
            {
                UserName = userName,
                EnabledAt = DateTime.UtcNow,
                SupportEmail = emailSettings["SupportEmail"] ?? "support@blessedrsi.com"
            };

            var result = await _fluentEmail
                .To(email)
                .Subject("BlessedRSI - Two-Factor Authentication Enabled")
                .UsingTemplate(GetTwoFactorEnabledTemplate(), model)
                .SendAsync();

            if (result.Successful)
            {
                _logger.LogInformation("2FA enabled notification sent to {Email}", email);
                return true;
            }
            else
            {
                _logger.LogError("Failed to send 2FA enabled notification to {Email}: {Errors}", 
                    email, string.Join(", ", result.ErrorMessages));
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending 2FA enabled notification to {Email}", email);
            return false;
        }
    }

    public async Task<bool> SendTwoFactorDisabledNotificationAsync(string email, string userName, string? ipAddress = null)
    {
        try
        {
            var emailSettings = _configuration.GetSection("EmailSettings");
            
            var model = new
            {
                UserName = userName,
                DisabledAt = DateTime.UtcNow,
                IpAddress = ipAddress ?? "Unknown",
                SupportEmail = emailSettings["SupportEmail"] ?? "support@blessedrsi.com"
            };

            var result = await _fluentEmail
                .To(email)
                .Subject("BlessedRSI - Two-Factor Authentication Disabled")
                .UsingTemplate(GetTwoFactorDisabledTemplate(), model)
                .SendAsync();

            if (result.Successful)
            {
                _logger.LogInformation("2FA disabled notification sent to {Email}", email);
                return true;
            }
            else
            {
                _logger.LogError("Failed to send 2FA disabled notification to {Email}: {Errors}", 
                    email, string.Join(", ", result.ErrorMessages));
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending 2FA disabled notification to {Email}", email);
            return false;
        }
    }

    public async Task<bool> SendSuspiciousActivityAlertAsync(string email, string userName, string activity, string? ipAddress = null)
    {
        try
        {
            var emailSettings = _configuration.GetSection("EmailSettings");
            
            var model = new
            {
                UserName = userName,
                Activity = activity,
                Timestamp = DateTime.UtcNow,
                IpAddress = ipAddress ?? "Unknown",
                SupportEmail = emailSettings["SupportEmail"] ?? "support@blessedrsi.com"
            };

            var result = await _fluentEmail
                .To(email)
                .Subject("BlessedRSI - Security Alert")
                .UsingTemplate(GetSuspiciousActivityTemplate(), model)
                .SendAsync();

            if (result.Successful)
            {
                _logger.LogInformation("Suspicious activity alert sent to {Email}", email);
                return true;
            }
            else
            {
                _logger.LogError("Failed to send suspicious activity alert to {Email}: {Errors}", 
                    email, string.Join(", ", result.ErrorMessages));
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending suspicious activity alert to {Email}", email);
            return false;
        }
    }

    private static string GetActionDescription(TwoFactorCodeType codeType) => codeType switch
    {
        TwoFactorCodeType.Login => "Sign In",
        TwoFactorCodeType.EmailVerification => "Email Verification",
        TwoFactorCodeType.PasswordReset => "Password Reset",
        TwoFactorCodeType.EmailChange => "Email Change",
        TwoFactorCodeType.SecurityAction => "Security Action",
        _ => "Account Verification"
    };

    private static int GetCodeExpirationMinutes(TwoFactorCodeType codeType) => codeType switch
    {
        TwoFactorCodeType.Login => 10,
        TwoFactorCodeType.EmailVerification => 30,
        TwoFactorCodeType.PasswordReset => 15,
        TwoFactorCodeType.EmailChange => 30,
        TwoFactorCodeType.SecurityAction => 15,
        _ => 15
    };

    private static string GetTwoFactorCodeTemplate() => @"
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>BlessedRSI - Verification Code</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #0d6efd, #0b5ed7); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
        .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
        .code-box { background: white; border: 2px dashed #0d6efd; padding: 20px; text-align: center; margin: 20px 0; border-radius: 5px; }
        .code { font-size: 32px; font-weight: bold; color: #0d6efd; letter-spacing: 5px; }
        .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 12px; }
        .verse { font-style: italic; color: #6c757d; text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class='header'>
        <h1>📊 BlessedRSI</h1>
        <p>Faith-Based Investment Platform</p>
    </div>
    
    <div class='content'>
        <h2>Hello @Model.UserName,</h2>
        
        <p>You requested a verification code for: <strong>@Model.Action</strong></p>
        
        <div class='code-box'>
            <div class='code'>@Model.VerificationCode</div>
            <p style='margin: 10px 0 0 0; color: #666;'>Enter this code to continue</p>
        </div>
        
        <div class='warning'>
            <strong>⚠️ Security Notice:</strong>
            <ul style='margin: 10px 0 0 0;'>
                <li>This code expires at @Model.ExpiresAt.ToString(""MMM dd, yyyy 'at' h:mm tt"")</li>
                <li>Request originated from IP: @Model.IpAddress</li>
                <li>Never share this code with anyone</li>
                <li>BlessedRSI will never ask for this code via phone or email</li>
            </ul>
        </div>
        
        <p>If you didn't request this code, please contact our support team immediately at <a href='mailto:@Model.SupportEmail'>@Model.SupportEmail</a>.</p>
        
        <div class='verse'>
            ""The simple believe anything, but the prudent give thought to their steps."" - Proverbs 14:15
        </div>
    </div>
    
    <div class='footer'>
        <p>© 2024 BlessedRSI - Built with faith and technology</p>
        <p>This is an automated message. Please do not reply to this email.</p>
    </div>
</body>
</html>";

    private static string GetBackupCodesTemplate() => @"
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>BlessedRSI - Backup Codes</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #0d6efd, #0b5ed7); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
        .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
        .codes-box { background: white; border: 2px solid #28a745; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .code-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0; }
        .backup-code { background: #f8f9fa; padding: 10px; text-align: center; font-family: monospace; font-size: 16px; font-weight: bold; border: 1px solid #dee2e6; border-radius: 3px; }
        .warning { background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; margin: 20px 0; color: #721c24; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 12px; }
        .verse { font-style: italic; color: #6c757d; text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class='header'>
        <h1>📊 BlessedRSI</h1>
        <p>Your Two-Factor Authentication Backup Codes</p>
    </div>
    
    <div class='content'>
        <h2>Hello @Model.UserName,</h2>
        
        <p>Your new backup codes have been generated. These codes can be used to access your account if you lose access to your email.</p>
        
        <div class='codes-box'>
            <h3 style='margin-top: 0; color: #28a745;'>🔐 Backup Codes</h3>
            <div class='code-grid'>
                @foreach (var code in Model.BackupCodes)
                {
                    <div class='backup-code'>@code</div>
                }
            </div>
            <p style='margin-bottom: 0; font-size: 14px; color: #666;'>Generated: @Model.GeneratedAt.ToString(""MMM dd, yyyy 'at' h:mm tt"")</p>
        </div>
        
        <div class='warning'>
            <strong>🚨 IMPORTANT SECURITY INSTRUCTIONS:</strong>
            <ul style='margin: 10px 0 0 0;'>
                <li><strong>Save these codes securely</strong> - Store them in a password manager or secure location</li>
                <li><strong>Each code can only be used once</strong> - They will be invalidated after use</li>
                <li><strong>Keep them private</strong> - Never share these codes with anyone</li>
                <li><strong>Generate new codes</strong> if you suspect they've been compromised</li>
                <li><strong>Delete this email</strong> after saving the codes securely</li>
            </ul>
        </div>
        
        <p>You can generate new backup codes at any time from your security settings.</p>
        
        <div class='verse'>
            ""Above all else, guard your heart, for everything you do flows from it."" - Proverbs 4:23
        </div>
    </div>
    
    <div class='footer'>
        <p>© 2024 BlessedRSI - Built with faith and technology</p>
        <p>This is an automated message. Please do not reply to this email.</p>
    </div>
</body>
</html>";

    private static string GetTwoFactorEnabledTemplate() => @"
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>BlessedRSI - Two-Factor Authentication Enabled</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
        .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
        .success-box { background: #d4edda; border: 1px solid #c3e6cb; padding: 20px; border-radius: 5px; margin: 20px 0; color: #155724; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 12px; }
        .verse { font-style: italic; color: #6c757d; text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class='header'>
        <h1>🛡️ Security Enhanced</h1>
        <p>Two-Factor Authentication Activated</p>
    </div>
    
    <div class='content'>
        <h2>Congratulations @Model.UserName!</h2>
        
        <div class='success-box'>
            <h3 style='margin-top: 0;'>✅ Two-Factor Authentication is now enabled</h3>
            <p>Your account security has been significantly enhanced. From now on, you'll need to verify your identity with a code sent to your email when signing in.</p>
            <p><strong>Enabled on:</strong> @Model.EnabledAt.ToString(""MMM dd, yyyy 'at' h:mm tt"")</p>
        </div>
        
        <h3>What this means for you:</h3>
        <ul>
            <li>🔒 <strong>Enhanced Security:</strong> Your account is now protected against unauthorized access</li>
            <li>📧 <strong>Email Verification:</strong> You'll receive a code via email for each sign-in</li>
            <li>🔑 <strong>Backup Codes:</strong> Use your backup codes if you lose email access</li>
            <li>⚙️ <strong>Manage Settings:</strong> You can modify 2FA settings in your security preferences</li>
        </ul>
        
        <p>If you didn't enable this feature, please contact support immediately at <a href='mailto:@Model.SupportEmail'>@Model.SupportEmail</a>.</p>
        
        <div class='verse'>
            ""The prudent see danger and take refuge, but the simple keep going and pay the penalty."" - Proverbs 22:3
        </div>
    </div>
    
    <div class='footer'>
        <p>© 2024 BlessedRSI - Built with faith and technology</p>
    </div>
</body>
</html>";

    private static string GetTwoFactorDisabledTemplate() => @"
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>BlessedRSI - Two-Factor Authentication Disabled</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #ffc107, #fd7e14); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
        .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
        .warning-box { background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 5px; margin: 20px 0; color: #856404; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class='header'>
        <h1>⚠️ Security Notice</h1>
        <p>Two-Factor Authentication Disabled</p>
    </div>
    
    <div class='content'>
        <h2>Hello @Model.UserName,</h2>
        
        <div class='warning-box'>
            <h3 style='margin-top: 0;'>🔓 Two-Factor Authentication has been disabled</h3>
            <p>Your account security level has been reduced. Your account is now protected only by your password.</p>
            <p><strong>Disabled on:</strong> @Model.DisabledAt.ToString(""MMM dd, yyyy 'at' h:mm tt"")</p>
            <p><strong>From IP:</strong> @Model.IpAddress</p>
        </div>
        
        <p>If you didn't disable two-factor authentication, please:</p>
        <ol>
            <li>Change your password immediately</li>
            <li>Re-enable two-factor authentication</li>
            <li>Contact our support team at <a href='mailto:@Model.SupportEmail'>@Model.SupportEmail</a></li>
        </ol>
        
        <p>We strongly recommend keeping two-factor authentication enabled for maximum account security.</p>
    </div>
    
    <div class='footer'>
        <p>© 2024 BlessedRSI - Built with faith and technology</p>
    </div>
</body>
</html>";

    private static string GetSuspiciousActivityTemplate() => @"
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>BlessedRSI - Security Alert</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #dc3545, #c82333); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
        .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
        .alert-box { background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 5px; margin: 20px 0; color: #721c24; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class='header'>
        <h1>🚨 Security Alert</h1>
        <p>Suspicious Activity Detected</p>
    </div>
    
    <div class='content'>
        <h2>Hello @Model.UserName,</h2>
        
        <div class='alert-box'>
            <h3 style='margin-top: 0;'>⚠️ Suspicious Activity Detected</h3>
            <p><strong>Activity:</strong> @Model.Activity</p>
            <p><strong>Time:</strong> @Model.Timestamp.ToString(""MMM dd, yyyy 'at' h:mm tt"")</p>
            <p><strong>IP Address:</strong> @Model.IpAddress</p>
        </div>
        
        <p>If this was you, no action is needed. If you don't recognize this activity:</p>
        <ol>
            <li><strong>Change your password</strong> immediately</li>
            <li><strong>Enable two-factor authentication</strong> if not already active</li>
            <li><strong>Review your account</strong> for any unauthorized changes</li>
            <li><strong>Contact support</strong> at <a href='mailto:@Model.SupportEmail'>@Model.SupportEmail</a></li>
        </ol>
    </div>
    
    <div class='footer'>
        <p>© 2024 BlessedRSI - Built with faith and technology</p>
    </div>
</body>
</html>";
}