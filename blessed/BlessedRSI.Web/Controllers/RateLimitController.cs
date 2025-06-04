using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
public class RateLimitController : ControllerBase
{
    private readonly RateLimitService _rateLimitService;
    private readonly ILogger<RateLimitController> _logger;

    public RateLimitController(RateLimitService rateLimitService, ILogger<RateLimitController> logger)
    {
        _rateLimitService = rateLimitService;
        _logger = logger;
    }

    [HttpGet("status")]
    public async Task<ActionResult> GetRateLimitStatus()
    {
        var ipAddress = GetClientIpAddress();
        var userId = User.FindFirst("sub")?.Value;
        var userTier = GetUserSubscriptionTier();
        
        try
        {
            var result = await _rateLimitService.CheckRateLimitAsync(
                ipAddress, 
                "/api/general", 
                userId, 
                userTier);

            return Ok(new
            {
                success = true,
                data = new
                {
                    ipAddress = ipAddress,
                    userId = userId ?? "anonymous",
                    subscriptionTier = userTier?.ToString() ?? "None",
                    rateLimitInfo = new
                    {
                        requestLimit = result.RequestLimit,
                        requestsRemaining = result.RequestsRemaining,
                        windowSize = result.WindowSize,
                        windowReset = result.WindowReset,
                        isAllowed = result.IsAllowed
                    },
                    limits = GetUserLimits(userTier)
                },
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking rate limit status for IP {IpAddress}", ipAddress);
            return StatusCode(500, new
            {
                success = false,
                message = "Error checking rate limit status"
            });
        }
    }

    [HttpGet("info")]
    public ActionResult GetRateLimitInfo()
    {
        var userTier = GetUserSubscriptionTier();
        var isAuthenticated = User.Identity?.IsAuthenticated == true;

        return Ok(new
        {
            success = true,
            data = new
            {
                isAuthenticated = isAuthenticated,
                subscriptionTier = userTier?.ToString() ?? "None",
                limits = GetUserLimits(userTier),
                rateLimitingEnabled = true,
                endpoints = new
                {
                    general = new { windowSize = "1 hour", description = "General API requests" },
                    authentication = new { windowSize = "1 minute", description = "Login/registration requests" },
                    registration = new { windowSize = "1 hour", description = "Account registration" },
                    passwordReset = new { windowSize = "1 hour", description = "Password reset requests" }
                }
            },
            timestamp = DateTime.UtcNow
        });
    }

    [HttpPost("test")]
    [Authorize]
    public async Task<ActionResult> TestRateLimit()
    {
        var ipAddress = GetClientIpAddress();
        var userId = User.FindFirst("sub")?.Value;
        var userTier = GetUserSubscriptionTier();

        var result = await _rateLimitService.CheckRateLimitAsync(
            ipAddress, 
            "/api/test", 
            userId, 
            userTier);

        return Ok(new
        {
            success = true,
            message = "Rate limit test completed",
            data = new
            {
                testTime = DateTime.UtcNow,
                rateLimitResult = new
                {
                    isAllowed = result.IsAllowed,
                    requestsRemaining = result.RequestsRemaining,
                    requestLimit = result.RequestLimit,
                    windowReset = result.WindowReset,
                    message = result.Message
                }
            }
        });
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

    private SubscriptionTier? GetUserSubscriptionTier()
    {
        if (User.Identity?.IsAuthenticated == true)
        {
            var tierClaim = User.FindFirst("subscription_tier")?.Value;
            if (!string.IsNullOrEmpty(tierClaim) && 
                Enum.TryParse<SubscriptionTier>(tierClaim, out var tier))
            {
                return tier;
            }
        }
        return null;
    }

    private object GetUserLimits(SubscriptionTier? tier)
    {
        if (!tier.HasValue)
        {
            return new
            {
                general = new { requestsPerHour = 50, requestsPerMinute = 10 },
                authentication = new { requestsPerMinute = 5 },
                registration = new { requestsPerHour = 3 },
                passwordReset = new { requestsPerHour = 2 },
                description = "Anonymous user limits"
            };
        }

        return tier.Value switch
        {
            SubscriptionTier.Sparrow => new
            {
                general = new { requestsPerHour = 100 },
                authentication = new { requestsPerMinute = 5 },
                registration = new { requestsPerHour = 3 },
                passwordReset = new { requestsPerHour = 2 },
                description = "Sparrow tier limits - \"Consider the sparrows...\" (Luke 12:24)"
            },
            SubscriptionTier.Lion => new
            {
                general = new { requestsPerHour = 1000 },
                authentication = new { requestsPerMinute = 10 },
                registration = new { requestsPerHour = 5 },
                passwordReset = new { requestsPerHour = 3 },
                description = "Lion tier limits - \"Bold as a lion\" (Proverbs 28:1)"
            },
            SubscriptionTier.Eagle => new
            {
                general = new { requestsPerHour = 5000 },
                authentication = new { requestsPerMinute = 15 },
                registration = new { requestsPerHour = 10 },
                passwordReset = new { requestsPerHour = 5 },
                description = "Eagle tier limits - \"Soar on wings like eagles\" (Isaiah 40:31)"
            },
            SubscriptionTier.Shepherd => new
            {
                general = new { requestsPerHour = 10000 },
                authentication = new { requestsPerMinute = 20 },
                registration = new { requestsPerHour = 20 },
                passwordReset = new { requestsPerHour = 10 },
                description = "Shepherd tier limits - Ultimate leadership tier"
            },
            _ => new
            {
                general = new { requestsPerHour = 100 },
                description = "Default limits"
            }
        };
    }
}