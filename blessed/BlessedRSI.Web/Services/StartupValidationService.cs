using Microsoft.Extensions.Options;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class StartupValidationService : IHostedService
{
    private readonly EnhancedEmailService _emailService;
    private readonly EmailConfiguration _emailConfig;
    private readonly ILogger<StartupValidationService> _logger;
    private readonly IServiceProvider _serviceProvider;

    public StartupValidationService(
        IServiceProvider serviceProvider,
        ILogger<StartupValidationService> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
        
        // Get services from scope since this is a singleton
        using var scope = serviceProvider.CreateScope();
        _emailService = scope.ServiceProvider.GetRequiredService<EnhancedEmailService>();
        _emailConfig = scope.ServiceProvider.GetRequiredService<IOptions<EmailConfiguration>>().Value;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting application validation checks...");

        try
        {
            await ValidateEmailConfigurationAsync();
            await ValidateDatabaseConnectionAsync();
            await ValidateRequiredConfigurationAsync();
            
            _logger.LogInformation("✅ All startup validation checks passed successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Startup validation failed - some features may not work correctly");
            // Don't throw - allow app to start but log warnings
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Startup validation service stopped");
        return Task.CompletedTask;
    }

    private async Task ValidateEmailConfigurationAsync()
    {
        _logger.LogInformation("Validating email configuration...");

        try
        {
            using var scope = _serviceProvider.CreateScope();
            var emailService = scope.ServiceProvider.GetRequiredService<EnhancedEmailService>();
            
            // Validate primary configuration
            var primaryValidation = await emailService.ValidateConfigurationAsync();
            
            if (primaryValidation.IsValid && primaryValidation.CanConnect)
            {
                _logger.LogInformation("✅ Primary email configuration is valid and SMTP connection successful");
            }
            else
            {
                _logger.LogWarning("⚠️ Primary email configuration issues: {Errors}", 
                    string.Join(", ", primaryValidation.Errors));
                
                // Try fallback if configured
                if (_emailConfig.FallbackConfiguration != null)
                {
                    var fallbackValidation = await emailService.ValidateConfigurationAsync(_emailConfig.FallbackConfiguration);
                    
                    if (fallbackValidation.IsValid && fallbackValidation.CanConnect)
                    {
                        _logger.LogInformation("✅ Fallback email configuration is valid and available");
                    }
                    else
                    {
                        _logger.LogError("❌ Both primary and fallback email configurations are invalid");
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ No fallback email configuration available");
                }
            }

            // Log configuration summary
            _logger.LogInformation("Email Configuration Summary:");
            _logger.LogInformation("  Primary SMTP: {Host}:{Port} (SSL: {SSL})", 
                _emailConfig.SmtpHost, _emailConfig.SmtpPort, _emailConfig.EnableSsl);
            _logger.LogInformation("  From: {FromEmail} ({FromName})", 
                _emailConfig.FromEmail, _emailConfig.FromName);
            _logger.LogInformation("  Fallback Available: {HasFallback}", 
                _emailConfig.FallbackConfiguration != null);
            _logger.LogInformation("  Max Retries: {MaxRetries}", _emailConfig.MaxRetries);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating email configuration");
            throw;
        }
    }

    private async Task ValidateDatabaseConnectionAsync()
    {
        _logger.LogInformation("Validating database connection...");

        try
        {
            using var scope = _serviceProvider.CreateScope();
            var context = scope.ServiceProvider.GetRequiredService<BlessedRSI.Web.Data.ApplicationDbContext>();
            
            // Test database connection
            await context.Database.CanConnectAsync();
            _logger.LogInformation("✅ Database connection successful");

            // Check for pending migrations
            var pendingMigrations = await context.Database.GetPendingMigrationsAsync();
            if (pendingMigrations.Any())
            {
                _logger.LogWarning("⚠️ There are {Count} pending database migrations: {Migrations}", 
                    pendingMigrations.Count(), string.Join(", ", pendingMigrations));
            }
            else
            {
                _logger.LogInformation("✅ Database schema is up to date");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Database connection failed");
            throw;
        }
    }

    private Task ValidateRequiredConfigurationAsync()
    {
        _logger.LogInformation("Validating required configuration settings...");

        var issues = new List<string>();

        // Check JWT settings
        try
        {
            using var scope = _serviceProvider.CreateScope();
            var config = scope.ServiceProvider.GetRequiredService<IConfiguration>();
            
            var jwtSecretKey = config["JwtSettings:SecretKey"];
            if (string.IsNullOrEmpty(jwtSecretKey) || jwtSecretKey.Length < 32)
            {
                issues.Add("JWT SecretKey is missing or too short (minimum 32 characters)");
            }

            var jwtIssuer = config["JwtSettings:Issuer"];
            if (string.IsNullOrEmpty(jwtIssuer))
            {
                issues.Add("JWT Issuer is not configured");
            }

            var jwtAudience = config["JwtSettings:Audience"];
            if (string.IsNullOrEmpty(jwtAudience))
            {
                issues.Add("JWT Audience is not configured");
            }

            // Check connection string
            var connectionString = config.GetConnectionString("DefaultConnection");
            if (string.IsNullOrEmpty(connectionString))
            {
                issues.Add("Database connection string is not configured");
            }

            // Check email settings that weren't validated above
            if (string.IsNullOrEmpty(_emailConfig.FromEmail))
            {
                issues.Add("Email FromEmail is not configured");
            }

            if (string.IsNullOrEmpty(_emailConfig.FromName))
            {
                issues.Add("Email FromName is not configured");
            }

            // Check rate limiting
            var rateLimitEnabled = config.GetValue<bool>("RateLimitSettings:Enabled");
            _logger.LogInformation("Rate limiting enabled: {Enabled}", rateLimitEnabled);

            // Check 2FA settings
            var twoFactorEnabled = config.GetValue<bool>("TwoFactorSettings:IsEnabled");
            _logger.LogInformation("Two-factor authentication enabled: {Enabled}", twoFactorEnabled);

            if (issues.Any())
            {
                foreach (var issue in issues)
                {
                    _logger.LogError("❌ Configuration issue: {Issue}", issue);
                }
                throw new InvalidOperationException($"Configuration validation failed: {string.Join(", ", issues)}");
            }
            else
            {
                _logger.LogInformation("✅ All required configuration settings are present");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating configuration");
            throw;
        }

        return Task.CompletedTask;
    }
}