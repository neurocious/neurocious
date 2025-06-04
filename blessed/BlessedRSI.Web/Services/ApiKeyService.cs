using Microsoft.EntityFrameworkCore;
using System.Security.Cryptography;
using System.Text;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class ApiKeyService
{
    private readonly ApplicationDbContext _context;
    private readonly ILogger<ApiKeyService> _logger;
    private const int API_KEY_LENGTH = 32;
    private const string API_KEY_PREFIX = "brsi_";

    public ApiKeyService(ApplicationDbContext context, ILogger<ApiKeyService> logger)
    {
        _context = context;
        _logger = logger;
    }

    public async Task<CreateApiKeyResponse> CreateApiKeyAsync(string userId, CreateApiKeyRequest request)
    {
        try
        {
            // Verify user exists and has appropriate subscription
            var user = await _context.Users.FindAsync(userId);
            if (user == null)
            {
                return new CreateApiKeyResponse
                {
                    Success = false,
                    Message = "User not found"
                };
            }

            // Check if user has Shepherd tier (required for API access)
            if (user.SubscriptionTier != SubscriptionTier.Shepherd)
            {
                return new CreateApiKeyResponse
                {
                    Success = false,
                    Message = "API access requires Shepherd subscription tier"
                };
            }

            // Check if user has reached maximum number of API keys (limit to 5)
            var existingKeysCount = await _context.UserApiKeys
                .CountAsync(k => k.UserId == userId && k.IsActive);

            if (existingKeysCount >= 5)
            {
                return new CreateApiKeyResponse
                {
                    Success = false,
                    Message = "Maximum number of API keys reached (5)"
                };
            }

            // Generate API key
            var (apiKey, keyHash, prefix) = GenerateApiKey();

            // Set default rate limit based on subscription tier
            var rateLimitPerHour = request.RequestLimitPerHour ?? GetDefaultRateLimit(user.SubscriptionTier);

            // Create API key entity
            var userApiKey = new UserApiKey
            {
                UserId = userId,
                Name = request.Name,
                Description = request.Description,
                ApiKeyHash = keyHash,
                Prefix = prefix,
                ExpiresAt = request.ExpiresAt,
                RequestLimitPerHour = rateLimitPerHour,
                CreatedAt = DateTime.UtcNow,
                IsActive = true
            };

            _context.UserApiKeys.Add(userApiKey);
            await _context.SaveChangesAsync();

            _logger.LogInformation("API key created for user {UserId} with name {KeyName}", userId, request.Name);

            return new CreateApiKeyResponse
            {
                Success = true,
                Message = "API key created successfully",
                ApiKey = apiKey,
                KeyInfo = MapToDto(userApiKey)
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating API key for user {UserId}", userId);
            return new CreateApiKeyResponse
            {
                Success = false,
                Message = "An error occurred while creating the API key"
            };
        }
    }

    public async Task<ApiKeyListResponse> GetUserApiKeysAsync(string userId)
    {
        try
        {
            var apiKeys = await _context.UserApiKeys
                .Where(k => k.UserId == userId)
                .OrderByDescending(k => k.CreatedAt)
                .ToListAsync();

            return new ApiKeyListResponse
            {
                Success = true,
                ApiKeys = apiKeys.Select(MapToDto).ToList()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving API keys for user {UserId}", userId);
            return new ApiKeyListResponse
            {
                Success = false,
                Message = "An error occurred while retrieving API keys"
            };
        }
    }

    public async Task<bool> UpdateApiKeyAsync(string userId, int keyId, UpdateApiKeyRequest request)
    {
        try
        {
            var apiKey = await _context.UserApiKeys
                .FirstOrDefaultAsync(k => k.Id == keyId && k.UserId == userId);

            if (apiKey == null)
            {
                return false;
            }

            // Update fields if provided
            if (!string.IsNullOrEmpty(request.Name))
                apiKey.Name = request.Name;

            if (request.Description != null)
                apiKey.Description = request.Description;

            if (request.IsActive.HasValue)
                apiKey.IsActive = request.IsActive.Value;

            if (request.ExpiresAt.HasValue)
                apiKey.ExpiresAt = request.ExpiresAt.Value;

            if (request.RequestLimitPerHour.HasValue)
                apiKey.RequestLimitPerHour = request.RequestLimitPerHour.Value;

            await _context.SaveChangesAsync();

            _logger.LogInformation("API key {KeyId} updated for user {UserId}", keyId, userId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating API key {KeyId} for user {UserId}", keyId, userId);
            return false;
        }
    }

    public async Task<bool> DeleteApiKeyAsync(string userId, int keyId)
    {
        try
        {
            var apiKey = await _context.UserApiKeys
                .FirstOrDefaultAsync(k => k.Id == keyId && k.UserId == userId);

            if (apiKey == null)
            {
                return false;
            }

            _context.UserApiKeys.Remove(apiKey);
            await _context.SaveChangesAsync();

            _logger.LogInformation("API key {KeyId} deleted for user {UserId}", keyId, userId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting API key {KeyId} for user {UserId}", keyId, userId);
            return false;
        }
    }

    public async Task<UserApiKey?> ValidateApiKeyAsync(string apiKey)
    {
        if (string.IsNullOrEmpty(apiKey) || !apiKey.StartsWith(API_KEY_PREFIX))
        {
            return null;
        }

        try
        {
            var keyHash = HashApiKey(apiKey);
            var userApiKey = await _context.UserApiKeys
                .Include(k => k.User)
                .FirstOrDefaultAsync(k => k.ApiKeyHash == keyHash && k.IsActive);

            if (userApiKey == null)
            {
                return null;
            }

            // Check if key is expired
            if (userApiKey.ExpiresAt.HasValue && userApiKey.ExpiresAt.Value < DateTime.UtcNow)
            {
                return null;
            }

            // Check if user still has appropriate subscription
            if (userApiKey.User.SubscriptionTier != SubscriptionTier.Shepherd)
            {
                return null;
            }

            // Update last used timestamp
            userApiKey.LastUsedAt = DateTime.UtcNow;
            userApiKey.RequestCount++;
            await _context.SaveChangesAsync();

            return userApiKey;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating API key");
            return null;
        }
    }

    public async Task<bool> CheckRateLimitAsync(int apiKeyId)
    {
        try
        {
            var oneHourAgo = DateTime.UtcNow.AddHours(-1);
            var requestsInLastHour = await _context.ApiKeyUsageLogs
                .CountAsync(log => log.ApiKeyId == apiKeyId && log.RequestedAt > oneHourAgo);

            var apiKey = await _context.UserApiKeys.FindAsync(apiKeyId);
            if (apiKey == null)
            {
                return false;
            }

            return requestsInLastHour < apiKey.RequestLimitPerHour;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking rate limit for API key {ApiKeyId}", apiKeyId);
            return false;
        }
    }

    public async Task LogApiUsageAsync(int apiKeyId, string endpoint, string method, int statusCode, string? ipAddress, string? userAgent, int responseTimeMs)
    {
        try
        {
            var usageLog = new ApiKeyUsageLog
            {
                ApiKeyId = apiKeyId,
                Endpoint = endpoint,
                Method = method,
                StatusCode = statusCode,
                IpAddress = ipAddress,
                UserAgent = userAgent?.Length > 500 ? userAgent[..500] : userAgent,
                RequestedAt = DateTime.UtcNow,
                ResponseTimeMs = responseTimeMs
            };

            _context.ApiKeyUsageLogs.Add(usageLog);
            await _context.SaveChangesAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error logging API usage for key {ApiKeyId}", apiKeyId);
        }
    }

    private (string apiKey, string keyHash, string prefix) GenerateApiKey()
    {
        using var rng = RandomNumberGenerator.Create();
        var keyBytes = new byte[API_KEY_LENGTH];
        rng.GetBytes(keyBytes);

        var keyString = Convert.ToBase64String(keyBytes)
            .Replace("+", "")
            .Replace("/", "")
            .Replace("=", "")[..API_KEY_LENGTH];

        var prefix = API_KEY_PREFIX + keyString[..8];
        var fullKey = API_KEY_PREFIX + keyString;
        var keyHash = HashApiKey(fullKey);

        return (fullKey, keyHash, prefix);
    }

    private string HashApiKey(string apiKey)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(apiKey));
        return Convert.ToBase64String(hashBytes);
    }

    private long GetDefaultRateLimit(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Shepherd => 10000, // High limit for Shepherd tier
            _ => 1000 // Default limit
        };
    }

    private UserApiKeyDto MapToDto(UserApiKey apiKey)
    {
        return new UserApiKeyDto
        {
            Id = apiKey.Id,
            Name = apiKey.Name,
            Description = apiKey.Description,
            Prefix = apiKey.Prefix,
            CreatedAt = apiKey.CreatedAt,
            LastUsedAt = apiKey.LastUsedAt,
            ExpiresAt = apiKey.ExpiresAt,
            IsActive = apiKey.IsActive,
            RequestCount = apiKey.RequestCount,
            RequestLimitPerHour = apiKey.RequestLimitPerHour
        };
    }
}