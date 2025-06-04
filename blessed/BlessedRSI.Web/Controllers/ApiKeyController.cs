using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize]
public class ApiKeyController : ControllerBase
{
    private readonly ApiKeyService _apiKeyService;
    private readonly ILogger<ApiKeyController> _logger;

    public ApiKeyController(ApiKeyService apiKeyService, ILogger<ApiKeyController> logger)
    {
        _apiKeyService = apiKeyService;
        _logger = logger;
    }

    [HttpGet]
    public async Task<ActionResult<ApiKeyListResponse>> GetApiKeys()
    {
        var userId = User.FindFirst("sub")?.Value;
        if (string.IsNullOrEmpty(userId))
        {
            return BadRequest(new ApiKeyListResponse
            {
                Success = false,
                Message = "User ID not found"
            });
        }

        var result = await _apiKeyService.GetUserApiKeysAsync(userId);
        return Ok(result);
    }

    [HttpPost]
    public async Task<ActionResult<CreateApiKeyResponse>> CreateApiKey([FromBody] CreateApiKeyRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new CreateApiKeyResponse
            {
                Success = false,
                Message = "Invalid request data",
                Errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList()
            });
        }

        var userId = User.FindFirst("sub")?.Value;
        if (string.IsNullOrEmpty(userId))
        {
            return BadRequest(new CreateApiKeyResponse
            {
                Success = false,
                Message = "User ID not found"
            });
        }

        var result = await _apiKeyService.CreateApiKeyAsync(userId, request);
        
        if (!result.Success)
        {
            return BadRequest(result);
        }

        _logger.LogInformation("API key created successfully for user {UserId}", userId);
        return Ok(result);
    }

    [HttpPut("{keyId}")]
    public async Task<ActionResult> UpdateApiKey(int keyId, [FromBody] UpdateApiKeyRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new
            {
                success = false,
                message = "Invalid request data",
                errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList()
            });
        }

        var userId = User.FindFirst("sub")?.Value;
        if (string.IsNullOrEmpty(userId))
        {
            return BadRequest(new { success = false, message = "User ID not found" });
        }

        var success = await _apiKeyService.UpdateApiKeyAsync(userId, keyId, request);
        
        if (!success)
        {
            return NotFound(new { success = false, message = "API key not found" });
        }

        return Ok(new { success = true, message = "API key updated successfully" });
    }

    [HttpDelete("{keyId}")]
    public async Task<ActionResult> DeleteApiKey(int keyId)
    {
        var userId = User.FindFirst("sub")?.Value;
        if (string.IsNullOrEmpty(userId))
        {
            return BadRequest(new { success = false, message = "User ID not found" });
        }

        var success = await _apiKeyService.DeleteApiKeyAsync(userId, keyId);
        
        if (!success)
        {
            return NotFound(new { success = false, message = "API key not found" });
        }

        _logger.LogInformation("API key {KeyId} deleted for user {UserId}", keyId, userId);
        return Ok(new { success = true, message = "API key deleted successfully" });
    }

    [HttpGet("{keyId}/usage")]
    public async Task<ActionResult> GetApiKeyUsage(int keyId, [FromQuery] DateTime? startDate, [FromQuery] DateTime? endDate)
    {
        var userId = User.FindFirst("sub")?.Value;
        if (string.IsNullOrEmpty(userId))
        {
            return BadRequest(new { success = false, message = "User ID not found" });
        }

        // This would be implemented to get usage statistics
        // For now, return a placeholder response
        return Ok(new
        {
            success = true,
            message = "Usage statistics retrieved successfully",
            data = new
            {
                keyId = keyId,
                totalRequests = 0,
                requestsToday = 0,
                avgResponseTime = 0,
                startDate = startDate ?? DateTime.UtcNow.AddDays(-30),
                endDate = endDate ?? DateTime.UtcNow
            }
        });
    }

    [HttpPost("test")]
    public async Task<ActionResult> TestApiKey()
    {
        var userId = User.FindFirst("sub")?.Value;
        var authType = User.FindFirst("auth_type")?.Value;
        var apiKeyId = User.FindFirst("api_key_id")?.Value;

        if (authType == "api_key")
        {
            return Ok(new
            {
                success = true,
                message = "API key authentication successful",
                data = new
                {
                    userId = userId,
                    apiKeyId = apiKeyId,
                    authenticatedAt = DateTime.UtcNow,
                    method = "API Key"
                }
            });
        }
        else
        {
            return Ok(new
            {
                success = true,
                message = "JWT token authentication successful",
                data = new
                {
                    userId = userId,
                    authenticatedAt = DateTime.UtcNow,
                    method = "JWT Token"
                }
            });
        }
    }
}