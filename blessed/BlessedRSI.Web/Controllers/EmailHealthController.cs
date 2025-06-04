using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize(Roles = "Admin")]
public class EmailHealthController : ControllerBase
{
    private readonly EnhancedEmailService _emailService;
    private readonly EmailConfiguration _emailConfig;
    private readonly ILogger<EmailHealthController> _logger;

    public EmailHealthController(
        EnhancedEmailService emailService,
        IOptions<EmailConfiguration> emailConfig,
        ILogger<EmailHealthController> logger)
    {
        _emailService = emailService;
        _emailConfig = emailConfig.Value;
        _logger = logger;
    }

    [HttpGet("status")]
    public async Task<ActionResult> GetEmailHealthStatus()
    {
        try
        {
            var primaryHealth = await _emailService.GetHealthStatusAsync(EmailProvider.Primary);
            var fallbackHealth = _emailConfig.FallbackConfiguration != null 
                ? await _emailService.GetHealthStatusAsync(EmailProvider.Fallback)
                : null;

            return Ok(new
            {
                success = true,
                data = new
                {
                    overall = new
                    {
                        isHealthy = primaryHealth.IsHealthy || (fallbackHealth?.IsHealthy == true),
                        primaryAvailable = primaryHealth.IsHealthy,
                        fallbackAvailable = fallbackHealth?.IsHealthy == true,
                        fallbackConfigured = fallbackHealth != null
                    },
                    providers = new
                    {
                        primary = MapHealthStatus(primaryHealth),
                        fallback = fallbackHealth != null ? MapHealthStatus(fallbackHealth) : null
                    },
                    configuration = new
                    {
                        smtpHost = _emailConfig.SmtpHost,
                        smtpPort = _emailConfig.SmtpPort,
                        fromEmail = _emailConfig.FromEmail,
                        sslEnabled = _emailConfig.EnableSsl,
                        authRequired = _emailConfig.RequireAuthentication,
                        maxRetries = _emailConfig.MaxRetries,
                        fallbackEnabled = _emailConfig.UseFallbackOnFailure
                    }
                },
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting email health status");
            return StatusCode(500, new
            {
                success = false,
                message = "Error retrieving email health status"
            });
        }
    }

    [HttpPost("test")]
    public async Task<ActionResult> TestEmailConfiguration([FromBody] EmailTestRequest? request = null)
    {
        try
        {
            var configToTest = request?.Configuration ?? _emailConfig;
            var validation = await _emailService.ValidateConfigurationAsync(configToTest);

            var response = new
            {
                success = validation.IsValid,
                data = new
                {
                    isValid = validation.IsValid,
                    canConnect = validation.CanConnect,
                    canAuthenticate = validation.CanAuthenticate,
                    canSendTest = validation.CanSendTest,
                    testDuration = validation.TestDuration,
                    errors = validation.Errors,
                    warnings = validation.Warnings
                },
                timestamp = DateTime.UtcNow
            };

            if (validation.IsValid)
            {
                _logger.LogInformation("Email configuration test passed");
                return Ok(response);
            }
            else
            {
                _logger.LogWarning("Email configuration test failed: {Errors}", string.Join(", ", validation.Errors));
                return BadRequest(response);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error testing email configuration");
            return StatusCode(500, new
            {
                success = false,
                message = "Error testing email configuration",
                error = ex.Message
            });
        }
    }

    [HttpPost("send-test")]
    public async Task<ActionResult> SendTestEmail([FromBody] SendTestEmailRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new
            {
                success = false,
                message = "Invalid request",
                errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage)
            });
        }

        try
        {
            var result = await _emailService.SendSecurityAlertAsync(
                request.ToEmail,
                "Test User",
                "This is a test email from BlessedRSI email system validation.",
                HttpContext.Connection.RemoteIpAddress?.ToString());

            return Ok(new
            {
                success = result.Success,
                data = new
                {
                    messageId = result.MessageId,
                    usedFallback = result.UsedFallback,
                    provider = result.Provider,
                    sendDuration = result.SendDuration,
                    attemptCount = result.AttemptCount,
                    errors = result.Errors
                },
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending test email to {Email}", request.ToEmail);
            return StatusCode(500, new
            {
                success = false,
                message = "Error sending test email",
                error = ex.Message
            });
        }
    }

    [HttpGet("metrics")]
    public async Task<ActionResult> GetEmailMetrics()
    {
        try
        {
            var primaryHealth = await _emailService.GetHealthStatusAsync(EmailProvider.Primary);
            var fallbackHealth = _emailConfig.FallbackConfiguration != null 
                ? await _emailService.GetHealthStatusAsync(EmailProvider.Fallback)
                : null;

            return Ok(new
            {
                success = true,
                data = new
                {
                    metrics = new
                    {
                        primaryProvider = new
                        {
                            uptime = CalculateUptime(primaryHealth),
                            averageResponseTime = primaryHealth.AverageResponseTime,
                            consecutiveFailures = primaryHealth.ConsecutiveFailures,
                            lastSuccessful = primaryHealth.LastSuccessful,
                            lastAttempt = primaryHealth.LastAttempt
                        },
                        fallbackProvider = fallbackHealth != null ? new
                        {
                            uptime = CalculateUptime(fallbackHealth),
                            averageResponseTime = fallbackHealth.AverageResponseTime,
                            consecutiveFailures = fallbackHealth.ConsecutiveFailures,
                            lastSuccessful = fallbackHealth.LastSuccessful,
                            lastAttempt = fallbackHealth.LastAttempt
                        } : null
                    },
                    biblical = new
                    {
                        verse = "\"Be faithful in small things because it is in them that your strength lies.\" - Mother Teresa",
                        reference = "Luke 16:10 - \"Whoever is faithful in very little is also faithful in much.\""
                    }
                },
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting email metrics");
            return StatusCode(500, new
            {
                success = false,
                message = "Error retrieving email metrics"
            });
        }
    }

    private object MapHealthStatus(EmailHealthStatus health)
    {
        return new
        {
            isHealthy = health.IsHealthy,
            provider = health.Provider,
            lastSuccessful = health.LastSuccessful,
            lastAttempt = health.LastAttempt,
            consecutiveFailures = health.ConsecutiveFailures,
            averageResponseTime = health.AverageResponseTime,
            lastError = health.LastError,
            uptime = CalculateUptime(health)
        };
    }

    private double CalculateUptime(EmailHealthStatus health)
    {
        if (health.LastAttempt == default || health.LastSuccessful == default)
            return 0.0;

        var totalTime = DateTime.UtcNow - health.LastSuccessful.AddDays(-30); // 30 day window
        var downTime = TimeSpan.FromMinutes(health.ConsecutiveFailures * 5); // Approximate downtime

        if (totalTime.TotalMinutes <= 0)
            return 100.0;

        var uptime = ((totalTime - downTime).TotalMinutes / totalTime.TotalMinutes) * 100;
        return Math.Max(0, Math.Min(100, uptime));
    }
}

public class EmailTestRequest
{
    public EmailConfiguration? Configuration { get; set; }
}

public class SendTestEmailRequest
{
    [Required]
    [EmailAddress]
    public string ToEmail { get; set; } = string.Empty;
}