using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize]
public class TwoFactorController : ControllerBase
{
    private readonly TwoFactorAuthService _twoFactorService;
    private readonly ILogger<TwoFactorController> _logger;

    public TwoFactorController(TwoFactorAuthService twoFactorService, ILogger<TwoFactorController> logger)
    {
        _twoFactorService = twoFactorService;
        _logger = logger;
    }

    [HttpGet("status")]
    public async Task<ActionResult<TwoFactorStatusResponse>> GetStatus()
    {
        var userId = User.FindFirst("sub")?.Value;
        if (string.IsNullOrEmpty(userId))
        {
            return BadRequest("User ID not found");
        }

        var status = await _twoFactorService.GetTwoFactorStatusAsync(userId);
        return Ok(status);
    }

    [HttpPost("send-setup-code")]
    public async Task<ActionResult<AuthResponse>> SendSetupCode()
    {
        try
        {
            var userId = User.FindFirst("sub")?.Value;
            if (string.IsNullOrEmpty(userId))
            {
                return BadRequest(new AuthResponse
                {
                    Success = false,
                    Message = "User ID not found"
                });
            }

            var user = await GetCurrentUserAsync(userId);
            if (user == null)
            {
                return BadRequest(new AuthResponse
                {
                    Success = false,
                    Message = "User not found"
                });
            }

            await _twoFactorService.GenerateAndSendTwoFactorCodeAsync(user, TwoFactorCodeType.EmailVerification);

            return Ok(new AuthResponse
            {
                Success = true,
                Message = "Verification code sent to your email"
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending setup code");
            return StatusCode(500, new AuthResponse
            {
                Success = false,
                Message = "An error occurred while sending the verification code"
            });
        }
    }

    [HttpPost("enable")]
    public async Task<ActionResult<TwoFactorResponse>> Enable([FromBody] EnableTwoFactorRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new TwoFactorResponse
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
            return BadRequest(new TwoFactorResponse
            {
                Success = false,
                Message = "User ID not found"
            });
        }

        var result = await _twoFactorService.EnableTwoFactorAsync(userId, request);
        
        if (!result.Success)
        {
            return BadRequest(result);
        }

        return Ok(result);
    }

    [HttpPost("disable")]
    public async Task<ActionResult<TwoFactorResponse>> Disable([FromBody] DisableTwoFactorRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new TwoFactorResponse
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
            return BadRequest(new TwoFactorResponse
            {
                Success = false,
                Message = "User ID not found"
            });
        }

        var result = await _twoFactorService.DisableTwoFactorAsync(userId, request);
        
        if (!result.Success)
        {
            return BadRequest(result);
        }

        return Ok(result);
    }

    [HttpPost("send-verification-code")]
    public async Task<ActionResult<AuthResponse>> SendVerificationCode()
    {
        try
        {
            var userId = User.FindFirst("sub")?.Value;
            if (string.IsNullOrEmpty(userId))
            {
                return BadRequest(new AuthResponse
                {
                    Success = false,
                    Message = "User ID not found"
                });
            }

            var user = await GetCurrentUserAsync(userId);
            if (user == null)
            {
                return BadRequest(new AuthResponse
                {
                    Success = false,
                    Message = "User not found"
                });
            }

            await _twoFactorService.GenerateAndSendTwoFactorCodeAsync(user, TwoFactorCodeType.SecurityAction);

            return Ok(new AuthResponse
            {
                Success = true,
                Message = "Verification code sent to your email"
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending verification code");
            return StatusCode(500, new AuthResponse
            {
                Success = false,
                Message = "An error occurred while sending the verification code"
            });
        }
    }

    [HttpPost("generate-backup-codes")]
    public async Task<ActionResult<BackupCodesResponse>> GenerateBackupCodes([FromBody] GenerateBackupCodesRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new BackupCodesResponse
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
            return BadRequest(new BackupCodesResponse
            {
                Success = false,
                Message = "User ID not found"
            });
        }

        var result = await _twoFactorService.GenerateNewBackupCodesAsync(userId, request);
        
        if (!result.Success)
        {
            return BadRequest(result);
        }

        return Ok(result);
    }

    [HttpPost("verify")]
    public async Task<ActionResult<TwoFactorResponse>> VerifyCode([FromBody] VerifyTwoFactorRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new TwoFactorResponse
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
            return BadRequest(new TwoFactorResponse
            {
                Success = false,
                Message = "User ID not found"
            });
        }

        try
        {
            var isValid = await _twoFactorService.VerifyTwoFactorCodeAsync(userId, request.Code, TwoFactorCodeType.Login);
            
            if (!isValid)
            {
                return BadRequest(new TwoFactorResponse
                {
                    Success = false,
                    Message = "Invalid or expired verification code"
                });
            }

            return Ok(new TwoFactorResponse
            {
                Success = true,
                Message = "Code verified successfully"
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error verifying 2FA code for user {UserId}", userId);
            return StatusCode(500, new TwoFactorResponse
            {
                Success = false,
                Message = "An error occurred while verifying the code"
            });
        }
    }

    [HttpPost("use-backup-code")]
    public async Task<ActionResult<TwoFactorResponse>> UseBackupCode([FromBody] UseTwoFactorBackupCodeRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new TwoFactorResponse
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
            return BadRequest(new TwoFactorResponse
            {
                Success = false,
                Message = "User ID not found"
            });
        }

        try
        {
            var isValid = await _twoFactorService.UseBackupCodeAsync(userId, request.BackupCode);
            
            if (!isValid)
            {
                return BadRequest(new TwoFactorResponse
                {
                    Success = false,
                    Message = "Invalid backup code"
                });
            }

            // Get remaining backup codes count
            var status = await _twoFactorService.GetTwoFactorStatusAsync(userId);

            return Ok(new TwoFactorResponse
            {
                Success = true,
                Message = "Backup code accepted",
                BackupCodeUsed = true,
                BackupCodesRemaining = status.BackupCodesRemaining
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error using backup code for user {UserId}", userId);
            return StatusCode(500, new TwoFactorResponse
            {
                Success = false,
                Message = "An error occurred while processing the backup code"
            });
        }
    }

    private async Task<ApplicationUser?> GetCurrentUserAsync(string userId)
    {
        try
        {
            // This would typically use UserManager or a service to get the user
            // For now, we'll return a placeholder implementation
            return await Task.FromResult<ApplicationUser?>(new ApplicationUser { Id = userId });
        }
        catch
        {
            return null;
        }
    }
}