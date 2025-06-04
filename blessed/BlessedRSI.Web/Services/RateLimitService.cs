using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class RateLimitService
{
    private readonly ApplicationDbContext _context;
    private readonly RateLimitSettings _settings;
    private readonly ILogger<RateLimitService> _logger;
    
    // In-memory cache for fast lookups (fallback if Redis is not available)
    private readonly ConcurrentDictionary<string, RateLimitEntry> _memoryCache = new();
    private readonly Timer _cleanupTimer;

    public RateLimitService(
        ApplicationDbContext context, 
        IOptions<RateLimitSettings> settings,
        ILogger<RateLimitService> logger)
    {
        _context = context;
        _settings = settings.Value;
        _logger = logger;
        
        // Setup cleanup timer
        _cleanupTimer = new Timer(CleanupExpiredEntries, null, 
            TimeSpan.FromMinutes(_settings.CleanupIntervalMinutes),
            TimeSpan.FromMinutes(_settings.CleanupIntervalMinutes));
    }

    public async Task<RateLimitResult> CheckRateLimitAsync(
        string ipAddress, 
        string endpoint, 
        string? userId = null, 
        SubscriptionTier? userTier = null)
    {
        if (!_settings.Enabled)
        {
            return new RateLimitResult 
            { 
                IsAllowed = true, 
                RequestsRemaining = int.MaxValue,
                Message = "Rate limiting disabled" 
            };
        }

        var configs = GetApplicableRateLimitConfigs(endpoint, userId != null, userTier);
        
        foreach (var config in configs)
        {
            var result = await CheckSpecificLimitAsync(ipAddress, endpoint, userId, config);
            if (!result.IsAllowed)
            {
                return result;
            }
        }

        return new RateLimitResult 
        { 
            IsAllowed = true, 
            RequestsRemaining = GetLowestRemainingRequests(configs, ipAddress, endpoint, userId),
            Message = "Request allowed" 
        };
    }

    private async Task<RateLimitResult> CheckSpecificLimitAsync(
        string ipAddress, 
        string endpoint, 
        string? userId, 
        RateLimitConfig config)
    {
        var key = BuildCacheKey(ipAddress, endpoint, userId, config.Name);
        var now = DateTime.UtcNow;
        var windowStart = GetWindowStart(now, config.WindowSize);
        
        // Try to get from memory cache first
        if (_memoryCache.TryGetValue(key, out var cachedEntry))
        {
            if (cachedEntry.WindowStart == windowStart)
            {
                if (cachedEntry.RequestCount >= config.RequestLimit)
                {
                    return new RateLimitResult
                    {
                        IsAllowed = false,
                        RequestsRemaining = 0,
                        RequestLimit = config.RequestLimit,
                        WindowSize = config.WindowSize,
                        WindowReset = windowStart.Add(config.WindowSize),
                        Message = $"Rate limit exceeded for {config.Name}. Limit: {config.RequestLimit} requests per {FormatTimeSpan(config.WindowSize)}"
                    };
                }
                
                // Update cache
                cachedEntry.RequestCount++;
                cachedEntry.LastRequest = now;
                
                // Update database asynchronously
                _ = Task.Run(() => UpdateDatabaseEntryAsync(key, cachedEntry));
                
                return new RateLimitResult
                {
                    IsAllowed = true,
                    RequestsRemaining = config.RequestLimit - cachedEntry.RequestCount,
                    RequestLimit = config.RequestLimit,
                    WindowSize = config.WindowSize,
                    WindowReset = windowStart.Add(config.WindowSize),
                    Message = "Request allowed"
                };
            }
        }

        // Get from database or create new entry
        var dbEntry = await GetOrCreateRateLimitEntryAsync(ipAddress, endpoint, userId, config.Name, windowStart);
        
        if (dbEntry.RequestCount >= config.RequestLimit)
        {
            return new RateLimitResult
            {
                IsAllowed = false,
                RequestsRemaining = 0,
                RequestLimit = config.RequestLimit,
                WindowSize = config.WindowSize,
                WindowReset = windowStart.Add(config.WindowSize),
                Message = $"Rate limit exceeded for {config.Name}. Limit: {config.RequestLimit} requests per {FormatTimeSpan(config.WindowSize)}"
            };
        }

        // Increment and update
        dbEntry.RequestCount++;
        dbEntry.LastRequest = now;
        
        // Update cache
        _memoryCache.AddOrUpdate(key, dbEntry, (k, v) => dbEntry);
        
        // Save to database
        await _context.SaveChangesAsync();

        return new RateLimitResult
        {
            IsAllowed = true,
            RequestsRemaining = config.RequestLimit - dbEntry.RequestCount,
            RequestLimit = config.RequestLimit,
            WindowSize = config.WindowSize,
            WindowReset = windowStart.Add(config.WindowSize),
            Message = "Request allowed"
        };
    }

    private List<RateLimitConfig> GetApplicableRateLimitConfigs(
        string endpoint, 
        bool isAuthenticated, 
        SubscriptionTier? userTier)
    {
        var configs = new List<RateLimitConfig>();

        // Authentication endpoints - strict limits
        if (endpoint.Contains("/auth/login") || endpoint.Contains("/auth/register"))
        {
            configs.Add(new RateLimitConfig
            {
                Name = "Auth",
                RequestLimit = _settings.AuthEndpointRequestsPerMinute,
                WindowSize = TimeSpan.FromMinutes(1),
                Endpoints = new[] { "/auth/login", "/auth/register" },
                ApplyToAuthenticated = true,
                ApplyToAnonymous = true
            });
        }

        if (endpoint.Contains("/auth/register"))
        {
            configs.Add(new RateLimitConfig
            {
                Name = "Registration",
                RequestLimit = _settings.RegistrationRequestsPerHour,
                WindowSize = TimeSpan.FromHours(1),
                Endpoints = new[] { "/auth/register" },
                ApplyToAuthenticated = false,
                ApplyToAnonymous = true
            });
        }

        if (endpoint.Contains("/auth/forgot-password") || endpoint.Contains("/auth/reset-password"))
        {
            configs.Add(new RateLimitConfig
            {
                Name = "PasswordReset",
                RequestLimit = _settings.PasswordResetRequestsPerHour,
                WindowSize = TimeSpan.FromHours(1),
                Endpoints = new[] { "/auth/forgot-password", "/auth/reset-password" },
                ApplyToAuthenticated = true,
                ApplyToAnonymous = true
            });
        }

        // General API limits based on authentication and subscription
        if (isAuthenticated && userTier.HasValue)
        {
            var limit = GetSubscriptionTierLimit(userTier.Value);
            configs.Add(new RateLimitConfig
            {
                Name = "Authenticated",
                RequestLimit = limit,
                WindowSize = TimeSpan.FromHours(1),
                Endpoints = new[] { "/api/" },
                RequiredTier = userTier,
                ApplyToAuthenticated = true,
                ApplyToAnonymous = false
            });
        }
        else if (!isAuthenticated)
        {
            // Anonymous user limits
            configs.Add(new RateLimitConfig
            {
                Name = "Anonymous",
                RequestLimit = _settings.AnonymousRequestsPerHour,
                WindowSize = TimeSpan.FromHours(1),
                Endpoints = new[] { "/api/" },
                ApplyToAuthenticated = false,
                ApplyToAnonymous = true
            });

            configs.Add(new RateLimitConfig
            {
                Name = "AnonymousPerMinute",
                RequestLimit = _settings.AnonymousRequestsPerMinute,
                WindowSize = TimeSpan.FromMinutes(1),
                Endpoints = new[] { "/api/" },
                ApplyToAuthenticated = false,
                ApplyToAnonymous = true
            });
        }

        return configs;
    }

    private int GetSubscriptionTierLimit(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => _settings.SparrowRequestsPerHour,
            SubscriptionTier.Lion => _settings.LionRequestsPerHour,
            SubscriptionTier.Eagle => _settings.EagleRequestsPerHour,
            SubscriptionTier.Shepherd => _settings.ShepherdRequestsPerHour,
            _ => _settings.SparrowRequestsPerHour
        };
    }

    private async Task<RateLimitEntry> GetOrCreateRateLimitEntryAsync(
        string ipAddress, 
        string endpoint, 
        string? userId, 
        string rateLimitType, 
        DateTime windowStart)
    {
        var entry = await _context.RateLimitEntries
            .FirstOrDefaultAsync(e => 
                e.IpAddress == ipAddress &&
                e.Endpoint == endpoint &&
                e.UserId == userId &&
                e.RateLimitType == rateLimitType &&
                e.WindowStart == windowStart);

        if (entry == null)
        {
            entry = new RateLimitEntry
            {
                IpAddress = ipAddress,
                Endpoint = endpoint,
                UserId = userId,
                RateLimitType = rateLimitType,
                WindowStart = windowStart,
                RequestCount = 0,
                LastRequest = DateTime.UtcNow
            };
            
            _context.RateLimitEntries.Add(entry);
        }

        return entry;
    }

    private async Task UpdateDatabaseEntryAsync(string key, RateLimitEntry entry)
    {
        try
        {
            using var scope = _context.Database.BeginTransaction();
            _context.Entry(entry).State = EntityState.Modified;
            await _context.SaveChangesAsync();
            await scope.CommitAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to update rate limit entry in database");
        }
    }

    private int GetLowestRemainingRequests(
        List<RateLimitConfig> configs, 
        string ipAddress, 
        string endpoint, 
        string? userId)
    {
        var minRemaining = int.MaxValue;
        
        foreach (var config in configs)
        {
            var key = BuildCacheKey(ipAddress, endpoint, userId, config.Name);
            if (_memoryCache.TryGetValue(key, out var entry))
            {
                var remaining = config.RequestLimit - entry.RequestCount;
                if (remaining < minRemaining)
                {
                    minRemaining = remaining;
                }
            }
        }
        
        return minRemaining == int.MaxValue ? 1000 : minRemaining;
    }

    private void CleanupExpiredEntries(object? state)
    {
        try
        {
            var cutoff = DateTime.UtcNow.AddHours(-24);
            
            // Clean memory cache
            var expiredKeys = _memoryCache
                .Where(kvp => kvp.Value.LastRequest < cutoff)
                .Select(kvp => kvp.Key)
                .ToList();
            
            foreach (var key in expiredKeys)
            {
                _memoryCache.TryRemove(key, out _);
            }
            
            // Clean database entries
            _ = Task.Run(async () =>
            {
                try
                {
                    var expiredEntries = await _context.RateLimitEntries
                        .Where(e => e.LastRequest < cutoff)
                        .ToListAsync();
                    
                    _context.RateLimitEntries.RemoveRange(expiredEntries);
                    await _context.SaveChangesAsync();
                    
                    _logger.LogInformation("Cleaned up {Count} expired rate limit entries", expiredEntries.Count);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to cleanup expired rate limit entries from database");
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to cleanup expired rate limit entries");
        }
    }

    private static string BuildCacheKey(string ipAddress, string endpoint, string? userId, string rateLimitType)
    {
        return $"rate_limit:{ipAddress}:{endpoint}:{userId ?? "anonymous"}:{rateLimitType}";
    }

    private static DateTime GetWindowStart(DateTime now, TimeSpan windowSize)
    {
        var ticks = now.Ticks / windowSize.Ticks;
        return new DateTime(ticks * windowSize.Ticks, DateTimeKind.Utc);
    }

    private static string FormatTimeSpan(TimeSpan timeSpan)
    {
        if (timeSpan.TotalDays >= 1)
            return $"{timeSpan.TotalDays:F0} day(s)";
        if (timeSpan.TotalHours >= 1)
            return $"{timeSpan.TotalHours:F0} hour(s)";
        if (timeSpan.TotalMinutes >= 1)
            return $"{timeSpan.TotalMinutes:F0} minute(s)";
        return $"{timeSpan.TotalSeconds:F0} second(s)";
    }

    public void Dispose()
    {
        _cleanupTimer?.Dispose();
    }
}