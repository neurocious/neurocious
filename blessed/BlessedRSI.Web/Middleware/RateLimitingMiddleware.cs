using System.Text.Json;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Middleware;

public class RateLimitingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<RateLimitingMiddleware> _logger;

    public RateLimitingMiddleware(RequestDelegate next, ILogger<RateLimitingMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context, RateLimitService rateLimitService)
    {
        // Skip rate limiting for static files and non-API routes
        if (ShouldSkipRateLimit(context.Request.Path))
        {
            await _next(context);
            return;
        }

        var ipAddress = GetClientIpAddress(context);
        var endpoint = GetNormalizedEndpoint(context.Request.Path);
        var userId = GetUserId(context);
        var userTier = GetUserSubscriptionTier(context);

        try
        {
            var rateLimitResult = await rateLimitService.CheckRateLimitAsync(
                ipAddress, 
                endpoint, 
                userId, 
                userTier);

            // Add rate limit headers
            AddRateLimitHeaders(context.Response, rateLimitResult);

            if (!rateLimitResult.IsAllowed)
            {
                _logger.LogWarning(
                    "Rate limit exceeded for IP {IpAddress}, User {UserId}, Endpoint {Endpoint}. {Message}",
                    ipAddress, userId ?? "anonymous", endpoint, rateLimitResult.Message);

                await WriteRateLimitExceededResponse(context, rateLimitResult);
                return;
            }

            _logger.LogDebug(
                "Rate limit check passed for IP {IpAddress}, User {UserId}, Endpoint {Endpoint}. Remaining: {Remaining}",
                ipAddress, userId ?? "anonymous", endpoint, rateLimitResult.RequestsRemaining);

            await _next(context);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in rate limiting middleware for IP {IpAddress}, Endpoint {Endpoint}", 
                ipAddress, endpoint);

            // Continue processing on rate limit service failure
            await _next(context);
        }
    }

    private static bool ShouldSkipRateLimit(PathString path)
    {
        var pathValue = path.Value?.ToLower();
        
        if (string.IsNullOrEmpty(pathValue))
            return true;

        // Skip static files
        if (pathValue.Contains(".css") || pathValue.Contains(".js") || 
            pathValue.Contains(".ico") || pathValue.Contains(".png") || 
            pathValue.Contains(".jpg") || pathValue.Contains(".gif") ||
            pathValue.Contains(".svg") || pathValue.Contains(".woff") ||
            pathValue.Contains(".ttf"))
        {
            return true;
        }

        // Skip Blazor signaling
        if (pathValue.Contains("/_blazor/") || pathValue.Contains("/communityhub"))
        {
            return true;
        }

        // Skip health checks
        if (pathValue.Contains("/health") || pathValue.Contains("/ping"))
        {
            return true;
        }

        return false;
    }

    private static string GetClientIpAddress(HttpContext context)
    {
        // Check for forwarded headers (load balancer, proxy)
        var forwardedFor = context.Request.Headers["X-Forwarded-For"].FirstOrDefault();
        if (!string.IsNullOrEmpty(forwardedFor))
        {
            // Take the first IP if multiple are present
            return forwardedFor.Split(',')[0].Trim();
        }

        var realIp = context.Request.Headers["X-Real-IP"].FirstOrDefault();
        if (!string.IsNullOrEmpty(realIp))
        {
            return realIp;
        }

        // Fallback to connection remote IP
        return context.Connection.RemoteIpAddress?.ToString() ?? "unknown";
    }

    private static string GetNormalizedEndpoint(PathString path)
    {
        var pathValue = path.Value?.ToLower() ?? "/";
        
        // Normalize common patterns to reduce cache key variations
        
        // Replace IDs with placeholder
        pathValue = System.Text.RegularExpressions.Regex.Replace(
            pathValue, @"/\d+(/|$)", "/[id]$1");
        
        // Replace GUIDs with placeholder
        pathValue = System.Text.RegularExpressions.Regex.Replace(
            pathValue, @"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(/|$)", 
            "/[guid]$1", System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        return pathValue;
    }

    private static string? GetUserId(HttpContext context)
    {
        if (context.User?.Identity?.IsAuthenticated == true)
        {
            return context.User.FindFirst("sub")?.Value ?? 
                   context.User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
        }
        return null;
    }

    private static SubscriptionTier? GetUserSubscriptionTier(HttpContext context)
    {
        if (context.User?.Identity?.IsAuthenticated == true)
        {
            var tierClaim = context.User.FindFirst("subscription_tier")?.Value;
            if (!string.IsNullOrEmpty(tierClaim) && 
                Enum.TryParse<SubscriptionTier>(tierClaim, out var tier))
            {
                return tier;
            }
        }
        return null;
    }

    private static void AddRateLimitHeaders(HttpResponse response, RateLimitResult rateLimitResult)
    {
        response.Headers.Add("X-RateLimit-Limit", rateLimitResult.RequestLimit.ToString());
        response.Headers.Add("X-RateLimit-Remaining", Math.Max(0, rateLimitResult.RequestsRemaining).ToString());
        response.Headers.Add("X-RateLimit-Reset", ((DateTimeOffset)rateLimitResult.WindowReset).ToUnixTimeSeconds().ToString());
        response.Headers.Add("X-RateLimit-Window", ((int)rateLimitResult.WindowSize.TotalSeconds).ToString());

        if (!rateLimitResult.IsAllowed)
        {
            response.Headers.Add("Retry-After", ((int)rateLimitResult.WindowSize.TotalSeconds).ToString());
        }
    }

    private static async Task WriteRateLimitExceededResponse(HttpContext context, RateLimitResult rateLimitResult)
    {
        context.Response.StatusCode = 429; // Too Many Requests
        context.Response.ContentType = "application/json";

        var response = new
        {
            success = false,
            message = rateLimitResult.Message,
            error = "RateLimitExceeded",
            details = new
            {
                limit = rateLimitResult.RequestLimit,
                remaining = rateLimitResult.RequestsRemaining,
                resetTime = rateLimitResult.WindowReset,
                windowSize = rateLimitResult.WindowSize.TotalSeconds
            },
            timestamp = DateTime.UtcNow,
            // Biblical encouragement for rate-limited users
            encouragement = "\"Wait for the Lord; be strong and take heart and wait for the Lord.\" - Psalm 27:14"
        };

        var json = JsonSerializer.Serialize(response, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });

        await context.Response.WriteAsync(json);
    }
}