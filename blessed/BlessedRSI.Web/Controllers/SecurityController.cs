using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize(Roles = "Admin")]
public class SecurityController : ControllerBase
{
    private readonly ApplicationDbContext _context;
    private readonly ContentSanitizationService _sanitizationService;
    private readonly ILogger<SecurityController> _logger;

    public SecurityController(
        ApplicationDbContext context,
        ContentSanitizationService sanitizationService,
        ILogger<SecurityController> logger)
    {
        _context = context;
        _sanitizationService = sanitizationService;
        _logger = logger;
    }

    [HttpGet("xss-attempts")]
    public async Task<ActionResult> GetXssAttempts([FromQuery] int days = 7)
    {
        try
        {
            var cutoffDate = DateTime.UtcNow.AddDays(-days);
            
            // Get security events related to XSS
            var xssEvents = await _context.SecurityEvents
                .Where(e => e.CreatedAt >= cutoffDate && 
                           (e.EventType == "XSS_ATTEMPT" || 
                            e.EventType == "CONTENT_SANITIZED" ||
                            e.EventType == "MALICIOUS_CONTENT_BLOCKED"))
                .Include(e => e.User)
                .OrderByDescending(e => e.CreatedAt)
                .Take(100)
                .ToListAsync();

            var summary = new
            {
                totalAttempts = xssEvents.Count,
                uniqueUsers = xssEvents.Select(e => e.UserId).Distinct().Count(),
                blocked = xssEvents.Count(e => e.EventType == "MALICIOUS_CONTENT_BLOCKED"),
                sanitized = xssEvents.Count(e => e.EventType == "CONTENT_SANITIZED"),
                attempts = xssEvents.Count(e => e.EventType == "XSS_ATTEMPT")
            };

            return Ok(new
            {
                success = true,
                data = new
                {
                    summary = summary,
                    events = xssEvents.Select(e => new
                    {
                        id = e.Id,
                        eventType = e.EventType,
                        userId = e.UserId,
                        userName = e.User?.UserName ?? "Unknown",
                        ipAddress = e.IpAddress,
                        userAgent = e.UserAgent,
                        details = e.Details,
                        createdAt = e.CreatedAt,
                        metadata = e.Metadata
                    }),
                    biblical = new
                    {
                        verse = "\"Above all else, guard your heart, for everything you do flows from it.\"",
                        reference = "Proverbs 4:23"
                    }
                },
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving XSS attempts");
            return StatusCode(500, new
            {
                success = false,
                message = "Error retrieving security data"
            });
        }
    }

    [HttpPost("test-content")]
    public ActionResult TestContentSecurity([FromBody] TestContentRequest request)
    {
        if (string.IsNullOrEmpty(request.Content))
        {
            return BadRequest(new
            {
                success = false,
                message = "Content is required"
            });
        }

        try
        {
            var result = _sanitizationService.SanitizeHtml(request.Content, request.ContentType);

            // Log security test
            _logger.LogInformation("Security test performed by admin for content type {ContentType}", request.ContentType);

            return Ok(new
            {
                success = true,
                data = new
                {
                    original = request.Content,
                    sanitized = result.SanitizedContent,
                    isValid = result.IsValid,
                    isModified = result.IsModified,
                    threatLevel = result.ThreatLevel.ToString(),
                    securityIssues = result.SecurityIssues,
                    isSafe = _sanitizationService.IsContentSafe(request.Content)
                },
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error testing content security");
            return StatusCode(500, new
            {
                success = false,
                message = "Error testing content security"
            });
        }
    }

    [HttpGet("security-metrics")]
    public async Task<ActionResult> GetSecurityMetrics()
    {
        try
        {
            var thirtyDaysAgo = DateTime.UtcNow.AddDays(-30);
            var sevenDaysAgo = DateTime.UtcNow.AddDays(-7);
            var today = DateTime.UtcNow.Date;

            // Get recent security events
            var recentEvents = await _context.SecurityEvents
                .Where(e => e.CreatedAt >= thirtyDaysAgo)
                .ToListAsync();

            // Calculate metrics
            var metrics = new
            {
                last30Days = new
                {
                    totalSecurityEvents = recentEvents.Count,
                    xssAttempts = recentEvents.Count(e => e.EventType.Contains("XSS")),
                    blockedAttempts = recentEvents.Count(e => e.EventType == "MALICIOUS_CONTENT_BLOCKED"),
                    uniqueThreats = recentEvents.Select(e => e.UserId).Distinct().Count()
                },
                last7Days = new
                {
                    totalSecurityEvents = recentEvents.Count(e => e.CreatedAt >= sevenDaysAgo),
                    xssAttempts = recentEvents.Count(e => e.CreatedAt >= sevenDaysAgo && e.EventType.Contains("XSS")),
                    blockedAttempts = recentEvents.Count(e => e.CreatedAt >= sevenDaysAgo && e.EventType == "MALICIOUS_CONTENT_BLOCKED")
                },
                today = new
                {
                    totalSecurityEvents = recentEvents.Count(e => e.CreatedAt >= today),
                    xssAttempts = recentEvents.Count(e => e.CreatedAt >= today && e.EventType.Contains("XSS"))
                },
                topThreats = recentEvents
                    .Where(e => !string.IsNullOrEmpty(e.UserId))
                    .GroupBy(e => e.UserId)
                    .OrderByDescending(g => g.Count())
                    .Take(5)
                    .Select(g => new
                    {
                        userId = g.Key,
                        eventCount = g.Count(),
                        lastEvent = g.Max(e => e.CreatedAt)
                    })
            };

            return Ok(new
            {
                success = true,
                data = new
                {
                    metrics = metrics,
                    protectionStatus = new
                    {
                        xssProtectionEnabled = true,
                        contentSanitizationActive = true,
                        rateLimitingActive = true,
                        twoFactorAvailable = true
                    },
                    biblical = new
                    {
                        verse = "\"The Lord will keep you from all harm—he will watch over your life.\"",
                        reference = "Psalm 121:7"
                    }
                },
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving security metrics");
            return StatusCode(500, new
            {
                success = false,
                message = "Error retrieving security metrics"
            });
        }
    }

    [HttpPost("log-security-event")]
    public async Task<ActionResult> LogSecurityEvent([FromBody] LogSecurityEventRequest request)
    {
        try
        {
            var securityEvent = new SecurityEvent
            {
                EventType = request.EventType,
                UserId = request.UserId,
                IpAddress = GetClientIpAddress(),
                UserAgent = Request.Headers.UserAgent.FirstOrDefault(),
                Details = request.Details,
                Metadata = request.Metadata ?? new Dictionary<string, object>(),
                CreatedAt = DateTime.UtcNow
            };

            _context.SecurityEvents.Add(securityEvent);
            await _context.SaveChangesAsync();

            _logger.LogWarning("Security event logged: {EventType} for user {UserId}", 
                request.EventType, request.UserId ?? "Anonymous");

            return Ok(new
            {
                success = true,
                message = "Security event logged",
                eventId = securityEvent.Id
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error logging security event");
            return StatusCode(500, new
            {
                success = false,
                message = "Error logging security event"
            });
        }
    }

    private string GetClientIpAddress()
    {
        var forwardedFor = HttpContext.Request.Headers["X-Forwarded-For"].FirstOrDefault();
        if (!string.IsNullOrEmpty(forwardedFor))
        {
            return forwardedFor.Split(',')[0].Trim();
        }

        var realIp = HttpContext.Request.Headers["X-Real-IP"].FirstOrDefault();
        if (!string.IsNullOrEmpty(realIp))
        {
            return realIp;
        }

        return HttpContext.Connection.RemoteIpAddress?.ToString() ?? "unknown";
    }
}

public class TestContentRequest
{
    public string Content { get; set; } = string.Empty;
    public ContentType ContentType { get; set; } = ContentType.General;
}

public class LogSecurityEventRequest
{
    public string EventType { get; set; } = string.Empty;
    public string? UserId { get; set; }
    public string Details { get; set; } = string.Empty;
    public Dictionary<string, object>? Metadata { get; set; }
}