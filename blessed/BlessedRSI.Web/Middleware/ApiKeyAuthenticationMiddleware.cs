using System.Security.Claims;
using BlessedRSI.Web.Services;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Middleware;

public class ApiKeyAuthenticationMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<ApiKeyAuthenticationMiddleware> _logger;

    public ApiKeyAuthenticationMiddleware(RequestDelegate next, ILogger<ApiKeyAuthenticationMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context, ApiKeyService apiKeyService)
    {
        // Only apply to API routes
        if (!context.Request.Path.StartsWithSegments("/api"))
        {
            await _next(context);
            return;
        }

        // Skip authentication routes and public endpoints
        var path = context.Request.Path.Value?.ToLower();
        if (path != null && (path.Contains("/auth/") || path.Contains("/public/")))
        {
            await _next(context);
            return;
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            // Check for API key in headers
            var apiKey = ExtractApiKey(context.Request);
            
            if (!string.IsNullOrEmpty(apiKey))
            {
                var userApiKey = await apiKeyService.ValidateApiKeyAsync(apiKey);
                
                if (userApiKey != null)
                {
                    // Check rate limiting
                    var withinRateLimit = await apiKeyService.CheckRateLimitAsync(userApiKey.Id);
                    
                    if (!withinRateLimit)
                    {
                        await WriteErrorResponse(context, 429, "Rate limit exceeded");
                        await LogUsage(apiKeyService, userApiKey.Id, context, 429, stopwatch.ElapsedMilliseconds);
                        return;
                    }

                    // Set user context for API key authentication
                    var claims = new List<Claim>
                    {
                        new(ClaimTypes.NameIdentifier, userApiKey.User.Id),
                        new("sub", userApiKey.User.Id),
                        new(ClaimTypes.Email, userApiKey.User.Email ?? ""),
                        new("api_key_id", userApiKey.Id.ToString()),
                        new("auth_type", "api_key"),
                        new("subscription_tier", userApiKey.User.SubscriptionTier.ToString())
                    };

                    var identity = new ClaimsIdentity(claims, "ApiKey");
                    context.User = new ClaimsPrincipal(identity);

                    await _next(context);
                    
                    // Log successful API usage
                    await LogUsage(apiKeyService, userApiKey.Id, context, context.Response.StatusCode, stopwatch.ElapsedMilliseconds);
                    return;
                }
                else
                {
                    await WriteErrorResponse(context, 401, "Invalid API key");
                    return;
                }
            }

            // No API key provided, continue with normal authentication
            await _next(context);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in API key authentication middleware");
            await WriteErrorResponse(context, 500, "Internal server error");
        }
    }

    private string? ExtractApiKey(HttpRequest request)
    {
        // Check Authorization header (Bearer token)
        var authHeader = request.Headers.Authorization.FirstOrDefault();
        if (!string.IsNullOrEmpty(authHeader) && authHeader.StartsWith("Bearer ") && authHeader.Contains("brsi_"))
        {
            return authHeader["Bearer ".Length..];
        }

        // Check X-API-Key header
        var apiKeyHeader = request.Headers["X-API-Key"].FirstOrDefault();
        if (!string.IsNullOrEmpty(apiKeyHeader))
        {
            return apiKeyHeader;
        }

        // Check query parameter
        var queryApiKey = request.Query["api_key"].FirstOrDefault();
        if (!string.IsNullOrEmpty(queryApiKey))
        {
            return queryApiKey;
        }

        return null;
    }

    private async Task WriteErrorResponse(HttpContext context, int statusCode, string message)
    {
        context.Response.StatusCode = statusCode;
        context.Response.ContentType = "application/json";

        var response = new
        {
            success = false,
            message = message,
            timestamp = DateTime.UtcNow
        };

        var json = System.Text.Json.JsonSerializer.Serialize(response);
        await context.Response.WriteAsync(json);
    }

    private async Task LogUsage(ApiKeyService apiKeyService, int apiKeyId, HttpContext context, int statusCode, long responseTimeMs)
    {
        try
        {
            var endpoint = $"{context.Request.Method} {context.Request.Path}";
            var ipAddress = context.Connection.RemoteIpAddress?.ToString();
            var userAgent = context.Request.Headers.UserAgent.FirstOrDefault();

            await apiKeyService.LogApiUsageAsync(
                apiKeyId,
                endpoint,
                context.Request.Method,
                statusCode,
                ipAddress,
                userAgent,
                (int)responseTimeMs
            );
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error logging API usage for key {ApiKeyId}", apiKeyId);
        }
    }
}