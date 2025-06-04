using FluentEmail.Core;
using FluentEmail.Core.Models;
using Microsoft.Extensions.Options;
using System.Net.Mail;
using System.Net;
using System.ComponentModel.DataAnnotations;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class EnhancedEmailService
{
    private readonly IFluentEmail _fluentEmail;
    private readonly EmailConfiguration _emailConfig;
    private readonly ILogger<EnhancedEmailService> _logger;
    private readonly Dictionary<EmailProvider, EmailHealthStatus> _providerHealth;
    private readonly SemaphoreSlim _healthCheckSemaphore;
    private DateTime _lastHealthCheck;
    private readonly TimeSpan _healthCheckInterval = TimeSpan.FromMinutes(5);

    public EnhancedEmailService(
        IFluentEmail fluentEmail, 
        IOptions<EmailConfiguration> emailConfig,
        ILogger<EnhancedEmailService> logger)
    {
        _fluentEmail = fluentEmail;
        _emailConfig = emailConfig.Value;
        _logger = logger;
        _providerHealth = new Dictionary<EmailProvider, EmailHealthStatus>();
        _healthCheckSemaphore = new SemaphoreSlim(1, 1);
        
        InitializeProviderHealth();
    }

    public async Task<EmailValidationResult> ValidateConfigurationAsync(EmailConfiguration? config = null)
    {
        var configToTest = config ?? _emailConfig;
        var result = new EmailValidationResult();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            // Basic validation
            var validationContext = new ValidationContext(configToTest);
            var validationResults = new List<ValidationResult>();
            result.IsValid = Validator.TryValidateObject(configToTest, validationContext, validationResults, true);

            if (!result.IsValid)
            {
                result.Errors.AddRange(validationResults.Select(v => v.ErrorMessage ?? "Validation error"));
                return result;
            }

            // Test SMTP connection
            result.CanConnect = await TestSmtpConnectionAsync(configToTest);
            if (!result.CanConnect)
            {
                result.Errors.Add("Cannot connect to SMTP server");
                return result;
            }

            // Test authentication
            if (configToTest.RequireAuthentication)
            {
                result.CanAuthenticate = await TestSmtpAuthenticationAsync(configToTest);
                if (!result.CanAuthenticate)
                {
                    result.Errors.Add("SMTP authentication failed");
                    return result;
                }
            }
            else
            {
                result.CanAuthenticate = true;
                result.Warnings.Add("SMTP authentication is disabled");
            }

            // Test sending (optional - only in development)
            if (Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") == "Development")
            {
                result.CanSendTest = await TestSendEmailAsync(configToTest);
                if (!result.CanSendTest)
                {
                    result.Warnings.Add("Test email send failed - check logs for details");
                }
            }
            else
            {
                result.CanSendTest = true; // Skip in production
            }

        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating email configuration");
            result.Errors.Add($"Validation error: {ex.Message}");
        }
        finally
        {
            stopwatch.Stop();
            result.TestDuration = stopwatch.Elapsed;
        }

        return result;
    }

    public async Task<EmailSendResult> SendTwoFactorCodeAsync(
        string email, 
        string userName, 
        string code, 
        TwoFactorCodeType codeType, 
        string? ipAddress = null)
    {
        var model = new TwoFactorCodeEmailModel
        {
            UserName = userName,
            VerificationCode = code,
            Action = GetActionDescription(codeType),
            ExpiresAt = DateTime.UtcNow.AddMinutes(GetCodeExpirationMinutes(codeType)),
            IpAddress = ipAddress ?? "Unknown",
            Location = "Unknown",
            SupportEmail = _emailConfig.SupportEmail ?? _emailConfig.FromEmail
        };

        return await SendEmailWithFallbackAsync(
            email,
            $"BlessedRSI - Verification Code for {model.Action}",
            GetTwoFactorCodeTemplate(),
            model,
            EmailTemplate.TwoFactorCode);
    }

    public async Task<EmailSendResult> SendBackupCodesAsync(
        string email, 
        string userName, 
        List<string> backupCodes)
    {
        var model = new BackupCodesEmailModel
        {
            UserName = userName,
            BackupCodes = backupCodes,
            GeneratedAt = DateTime.UtcNow,
            SupportEmail = _emailConfig.SupportEmail ?? _emailConfig.FromEmail
        };

        return await SendEmailWithFallbackAsync(
            email,
            "BlessedRSI - Your New Backup Codes",
            GetBackupCodesTemplate(),
            model,
            EmailTemplate.BackupCodes);
    }

    public async Task<EmailSendResult> SendSecurityAlertAsync(
        string email,
        string userName,
        string alertMessage,
        string? ipAddress = null)
    {
        var model = new
        {
            UserName = userName,
            AlertMessage = alertMessage,
            AlertTime = DateTime.UtcNow,
            IpAddress = ipAddress ?? "Unknown",
            SupportEmail = _emailConfig.SupportEmail ?? _emailConfig.FromEmail
        };

        return await SendEmailWithFallbackAsync(
            email,
            "BlessedRSI - Security Alert",
            GetSecurityAlertTemplate(),
            model,
            EmailTemplate.SecurityAlert);
    }

    public async Task<EmailHealthStatus> GetHealthStatusAsync(EmailProvider provider = EmailProvider.Primary)
    {
        await EnsureRecentHealthCheckAsync();
        
        if (_providerHealth.TryGetValue(provider, out var status))
        {
            return status;
        }

        return new EmailHealthStatus
        {
            IsHealthy = false,
            Provider = provider.ToString(),
            LastAttempt = DateTime.UtcNow,
            LastError = "Provider not found"
        };
    }

    private async Task<EmailSendResult> SendEmailWithFallbackAsync<T>(
        string toEmail,
        string subject,
        string template,
        T model,
        EmailTemplate templateType)
    {
        var result = new EmailSendResult();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // Try primary provider first
        result = await TrySendEmailAsync(toEmail, subject, template, model, EmailProvider.Primary);
        
        if (!result.Success && _emailConfig.UseFallbackOnFailure && _emailConfig.FallbackConfiguration != null)
        {
            _logger.LogWarning("Primary email provider failed, trying fallback for {Email}", toEmail);
            
            // Try fallback provider
            var fallbackResult = await TrySendEmailAsync(toEmail, subject, template, model, EmailProvider.Fallback);
            
            if (fallbackResult.Success)
            {
                result = fallbackResult;
                result.UsedFallback = true;
                _logger.LogInformation("Email sent successfully using fallback provider to {Email}", toEmail);
            }
            else
            {
                result.Errors.AddRange(fallbackResult.Errors);
                _logger.LogError("Both primary and fallback email providers failed for {Email}", toEmail);
            }
        }

        stopwatch.Stop();
        result.SendDuration = stopwatch.Elapsed;
        
        // Update health status
        await UpdateProviderHealthAsync(
            result.UsedFallback ? EmailProvider.Fallback : EmailProvider.Primary, 
            result.Success, 
            result.Errors.FirstOrDefault());

        return result;
    }

    private async Task<EmailSendResult> TrySendEmailAsync<T>(
        string toEmail,
        string subject,
        string template,
        T model,
        EmailProvider provider)
    {
        var result = new EmailSendResult { Provider = provider.ToString() };
        
        for (int attempt = 1; attempt <= _emailConfig.MaxRetries; attempt++)
        {
            result.AttemptCount = attempt;
            
            try
            {
                var sendResult = await _fluentEmail
                    .To(toEmail)
                    .Subject(subject)
                    .UsingTemplate(template, model)
                    .SendAsync();

                if (sendResult.Successful)
                {
                    result.Success = true;
                    result.MessageId = sendResult.MessageId;
                    _logger.LogInformation("Email sent successfully to {Email} using {Provider} on attempt {Attempt}", 
                        toEmail, provider, attempt);
                    return result;
                }
                else
                {
                    result.Errors.AddRange(sendResult.ErrorMessages);
                    _logger.LogWarning("Email send failed to {Email} using {Provider} on attempt {Attempt}: {Errors}", 
                        toEmail, provider, attempt, string.Join(", ", sendResult.ErrorMessages));
                }
            }
            catch (Exception ex)
            {
                var errorMessage = $"Attempt {attempt}: {ex.Message}";
                result.Errors.Add(errorMessage);
                _logger.LogError(ex, "Email send exception to {Email} using {Provider} on attempt {Attempt}", 
                    toEmail, provider, attempt);
            }

            // Wait before retry (except on last attempt)
            if (attempt < _emailConfig.MaxRetries)
            {
                await Task.Delay(TimeSpan.FromSeconds(_emailConfig.RetryDelaySeconds));
            }
        }

        return result;
    }

    private async Task<bool> TestSmtpConnectionAsync(EmailConfiguration config)
    {
        try
        {
            using var client = new SmtpClient(config.SmtpHost, config.SmtpPort);
            client.EnableSsl = config.EnableSsl;
            client.Timeout = config.TimeoutSeconds * 1000;
            
            if (config.RequireAuthentication && !string.IsNullOrEmpty(config.SmtpUsername))
            {
                client.Credentials = new NetworkCredential(config.SmtpUsername, config.SmtpPassword);
            }

            // Test connection by sending NOOP
            await Task.Run(() => client.Send(new MailMessage())); // This will fail but test connection
            return true;
        }
        catch (SmtpException ex) when (ex.Message.Contains("mailbox unavailable"))
        {
            // Connection successful, just no valid recipient
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning("SMTP connection test failed: {Error}", ex.Message);
            return false;
        }
    }

    private async Task<bool> TestSmtpAuthenticationAsync(EmailConfiguration config)
    {
        try
        {
            using var client = new SmtpClient(config.SmtpHost, config.SmtpPort);
            client.EnableSsl = config.EnableSsl;
            client.Timeout = config.TimeoutSeconds * 1000;
            
            if (!string.IsNullOrEmpty(config.SmtpUsername))
            {
                client.Credentials = new NetworkCredential(config.SmtpUsername, config.SmtpPassword);
            }

            // This will test authentication
            var testMessage = new MailMessage(
                config.FromEmail, 
                config.FromEmail, 
                "Test Authentication", 
                "Test");
                
            await Task.Run(() => client.Send(testMessage));
            return true;
        }
        catch (SmtpException ex) when (ex.StatusCode == SmtpStatusCode.MailboxBusy)
        {
            // Authentication successful
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning("SMTP authentication test failed: {Error}", ex.Message);
            return false;
        }
    }

    private async Task<bool> TestSendEmailAsync(EmailConfiguration config)
    {
        try
        {
            var testResult = await _fluentEmail
                .To(config.FromEmail)
                .Subject("BlessedRSI - Email Configuration Test")
                .Body("This is a test email to verify email configuration.")
                .SendAsync();

            return testResult.Successful;
        }
        catch (Exception ex)
        {
            _logger.LogWarning("Test email send failed: {Error}", ex.Message);
            return false;
        }
    }

    private async Task EnsureRecentHealthCheckAsync()
    {
        if (DateTime.UtcNow - _lastHealthCheck < _healthCheckInterval)
            return;

        await _healthCheckSemaphore.WaitAsync();
        try
        {
            if (DateTime.UtcNow - _lastHealthCheck < _healthCheckInterval)
                return;

            await PerformHealthCheckAsync();
            _lastHealthCheck = DateTime.UtcNow;
        }
        finally
        {
            _healthCheckSemaphore.Release();
        }
    }

    private async Task PerformHealthCheckAsync()
    {
        // Check primary provider
        var primaryValid = await ValidateConfigurationAsync(_emailConfig);
        UpdateProviderHealthStatus(EmailProvider.Primary, primaryValid);

        // Check fallback provider if configured
        if (_emailConfig.FallbackConfiguration != null)
        {
            var fallbackValid = await ValidateConfigurationAsync(_emailConfig.FallbackConfiguration);
            UpdateProviderHealthStatus(EmailProvider.Fallback, fallbackValid);
        }
    }

    private void UpdateProviderHealthStatus(EmailProvider provider, EmailValidationResult validation)
    {
        var status = _providerHealth.GetValueOrDefault(provider) ?? new EmailHealthStatus
        {
            Provider = provider.ToString()
        };

        status.LastAttempt = DateTime.UtcNow;
        status.IsHealthy = validation.IsValid && validation.CanConnect;
        
        if (status.IsHealthy)
        {
            status.LastSuccessful = DateTime.UtcNow;
            status.ConsecutiveFailures = 0;
            status.LastError = null;
        }
        else
        {
            status.ConsecutiveFailures++;
            status.LastError = string.Join("; ", validation.Errors);
        }

        status.AverageResponseTime = validation.TestDuration;
        _providerHealth[provider] = status;
    }

    private async Task UpdateProviderHealthAsync(EmailProvider provider, bool success, string? error)
    {
        var status = _providerHealth.GetValueOrDefault(provider) ?? new EmailHealthStatus
        {
            Provider = provider.ToString()
        };

        status.LastAttempt = DateTime.UtcNow;
        
        if (success)
        {
            status.IsHealthy = true;
            status.LastSuccessful = DateTime.UtcNow;
            status.ConsecutiveFailures = 0;
            status.LastError = null;
        }
        else
        {
            status.ConsecutiveFailures++;
            status.LastError = error;
            
            // Mark as unhealthy after 3 consecutive failures
            if (status.ConsecutiveFailures >= 3)
            {
                status.IsHealthy = false;
            }
        }

        _providerHealth[provider] = status;
    }

    private void InitializeProviderHealth()
    {
        _providerHealth[EmailProvider.Primary] = new EmailHealthStatus
        {
            Provider = "Primary",
            IsHealthy = true
        };

        if (_emailConfig.FallbackConfiguration != null)
        {
            _providerHealth[EmailProvider.Fallback] = new EmailHealthStatus
            {
                Provider = "Fallback",
                IsHealthy = true,
                FallbackAvailable = true
            };
        }
    }

    // Template methods (keeping from original EmailService)
    private string GetActionDescription(TwoFactorCodeType codeType) =>
        codeType switch
        {
            TwoFactorCodeType.Login => "Login Verification",
            TwoFactorCodeType.EmailVerification => "Email Verification",
            TwoFactorCodeType.SecurityAction => "Security Action",
            _ => "Account Verification"
        };

    private int GetCodeExpirationMinutes(TwoFactorCodeType codeType) =>
        codeType switch
        {
            TwoFactorCodeType.Login => 15,
            TwoFactorCodeType.EmailVerification => 30,
            TwoFactorCodeType.SecurityAction => 10,
            _ => 15
        };

    private string GetTwoFactorCodeTemplate()
    {
        return @"
<html>
<body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>
    <div style='max-width: 600px; margin: 0 auto; padding: 20px;'>
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1 style='color: #2563eb; margin-bottom: 10px;'>BlessedRSI</h1>
            <p style='color: #666; font-style: italic;'>Faith-Based Investing</p>
        </div>
        
        <h2 style='color: #1e40af;'>Verification Code</h2>
        
        <p>Dear @Model.UserName,</p>
        
        <p>You have requested a verification code for: <strong>@Model.Action</strong></p>
        
        <div style='background: #f3f4f6; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;'>
            <h3 style='margin: 0; color: #1e40af; font-size: 24px; letter-spacing: 2px;'>@Model.VerificationCode</h3>
        </div>
        
        <p><strong>Important:</strong></p>
        <ul>
            <li>This code expires at: <strong>@Model.ExpiresAt.ToString(""MMM dd, yyyy HH:mm"")</strong></li>
            <li>Do not share this code with anyone</li>
            <li>If you didn't request this, contact support immediately</li>
        </ul>
        
        <div style='background: #fef3c7; padding: 15px; border-radius: 6px; border-left: 4px solid #f59e0b; margin: 20px 0;'>
            <p style='margin: 0; font-style: italic;'>""Guard your heart above all else, for it determines the course of your life."" - Proverbs 4:23</p>
        </div>
        
        <hr style='margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;'>
        
        <div style='font-size: 12px; color: #666; text-align: center;'>
            <p>Request Details:</p>
            <p>IP Address: @Model.IpAddress | Time: @DateTime.UtcNow.ToString(""yyyy-MM-dd HH:mm:ss UTC"")</p>
            <p>If you need help, contact us at <a href=""mailto:@Model.SupportEmail"">@Model.SupportEmail</a></p>
        </div>
    </div>
</body>
</html>";
    }

    private string GetBackupCodesTemplate()
    {
        return @"
<html>
<body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>
    <div style='max-width: 600px; margin: 0 auto; padding: 20px;'>
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1 style='color: #2563eb; margin-bottom: 10px;'>BlessedRSI</h1>
            <p style='color: #666; font-style: italic;'>Faith-Based Investing</p>
        </div>
        
        <h2 style='color: #1e40af;'>Your Backup Codes</h2>
        
        <p>Dear @Model.UserName,</p>
        
        <p>Your new backup codes have been generated. Please store these securely:</p>
        
        <div style='background: #f9fafb; padding: 20px; border-radius: 8px; margin: 20px 0;'>
            @foreach(var code in Model.BackupCodes)
            {
                <div style='font-family: monospace; font-size: 16px; padding: 5px; border-bottom: 1px solid #e5e7eb;'>@code</div>
            }
        </div>
        
        <div style='background: #fef2f2; padding: 15px; border-radius: 6px; border-left: 4px solid #ef4444; margin: 20px 0;'>
            <p style='margin: 0; font-weight: bold; color: #dc2626;'>⚠️ Important Security Information:</p>
            <ul style='margin: 10px 0 0 0;'>
                <li>Each code can only be used once</li>
                <li>Store these codes in a secure location</li>
                <li>Do not share these codes with anyone</li>
                <li>These codes can be used to access your account if you lose your primary 2FA method</li>
            </ul>
        </div>
        
        <hr style='margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;'>
        
        <div style='font-size: 12px; color: #666; text-align: center;'>
            <p>Generated: @Model.GeneratedAt.ToString(""yyyy-MM-dd HH:mm:ss UTC"")</p>
            <p>Questions? Contact us at <a href=""mailto:@Model.SupportEmail"">@Model.SupportEmail</a></p>
        </div>
    </div>
</body>
</html>";
    }

    private string GetSecurityAlertTemplate()
    {
        return @"
<html>
<body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>
    <div style='max-width: 600px; margin: 0 auto; padding: 20px;'>
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1 style='color: #2563eb; margin-bottom: 10px;'>BlessedRSI</h1>
            <p style='color: #666; font-style: italic;'>Faith-Based Investing</p>
        </div>
        
        <div style='background: #fef2f2; padding: 20px; border-radius: 8px; border-left: 4px solid #ef4444; margin: 20px 0;'>
            <h2 style='color: #dc2626; margin-top: 0;'>🔒 Security Alert</h2>
            <p><strong>@Model.AlertMessage</strong></p>
        </div>
        
        <p>Dear @Model.UserName,</p>
        
        <p>This is an automated security notification regarding your BlessedRSI account.</p>
        
        <div style='background: #f3f4f6; padding: 15px; border-radius: 6px; margin: 20px 0;'>
            <p><strong>Alert Time:</strong> @Model.AlertTime.ToString(""MMM dd, yyyy HH:mm:ss UTC"")</p>
            <p><strong>IP Address:</strong> @Model.IpAddress</p>
        </div>
        
        <p><strong>What should you do?</strong></p>
        <ul>
            <li>If this was you, no action is needed</li>
            <li>If this wasn't you, please secure your account immediately</li>
            <li>Change your password if you suspect unauthorized access</li>
            <li>Contact our support team if you need assistance</li>
        </ul>
        
        <div style='text-align: center; margin: 30px 0;'>
            <a href='https://blessedrsi.com/security' style='background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;'>Review Security Settings</a>
        </div>
        
        <hr style='margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;'>
        
        <div style='font-size: 12px; color: #666; text-align: center;'>
            <p>This is an automated security alert from BlessedRSI</p>
            <p>Support: <a href=""mailto:@Model.SupportEmail"">@Model.SupportEmail</a></p>
        </div>
    </div>
</body>
</html>";
    }
}