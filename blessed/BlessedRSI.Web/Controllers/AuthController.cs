using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    private readonly AuthService _authService;
    private readonly ILogger<AuthController> _logger;

    public AuthController(AuthService authService, ILogger<AuthController> logger)
    {
        _authService = authService;
        _logger = logger;
    }

    [HttpPost("register")]
    public async Task<ActionResult<AuthResponse>> Register([FromBody] RegisterRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new AuthResponse
            {
                Success = false,
                Message = "Invalid request data",
                Errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList()
            });
        }

        var result = await _authService.RegisterAsync(request);
        
        if (!result.Success)
        {
            return BadRequest(result);
        }

        // Set refresh token in HTTP-only cookie
        SetRefreshTokenCookie(result.RefreshToken!);
        
        // Don't send refresh token in response body for security
        result.RefreshToken = null;

        return Ok(result);
    }

    [HttpPost("login")]
    public async Task<ActionResult<AuthResponse>> Login([FromBody] LoginRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new AuthResponse
            {
                Success = false,
                Message = "Invalid request data",
                Errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList()
            });
        }

        var result = await _authService.LoginAsync(request);
        
        if (!result.Success)
        {
            return BadRequest(result);
        }

        // Set refresh token in HTTP-only cookie
        SetRefreshTokenCookie(result.RefreshToken!);
        
        // Don't send refresh token in response body for security
        result.RefreshToken = null;

        return Ok(result);
    }

    [HttpPost("refresh")]
    public async Task<ActionResult<AuthResponse>> RefreshToken([FromBody] RefreshTokenRequest request)
    {
        // Try to get refresh token from cookie if not provided in body
        if (string.IsNullOrEmpty(request.RefreshToken))
        {
            request.RefreshToken = GetRefreshTokenFromCookie();
        }

        if (string.IsNullOrEmpty(request.RefreshToken))
        {
            return BadRequest(new AuthResponse
            {
                Success = false,
                Message = "Refresh token is required",
                Errors = new List<string> { "No refresh token provided" }
            });
        }

        var result = await _authService.RefreshTokenAsync(request);
        
        if (!result.Success)
        {
            // Clear invalid refresh token cookie
            ClearRefreshTokenCookie();
            return BadRequest(result);
        }

        // Set new refresh token in HTTP-only cookie
        SetRefreshTokenCookie(result.RefreshToken!);
        
        // Don't send refresh token in response body for security
        result.RefreshToken = null;

        return Ok(result);
    }

    [HttpPost("logout")]
    [Authorize]
    public async Task<ActionResult<AuthResponse>> Logout()
    {
        var userId = User.FindFirst("sub")?.Value;
        var refreshToken = GetRefreshTokenFromCookie();

        if (!string.IsNullOrEmpty(userId) && !string.IsNullOrEmpty(refreshToken))
        {
            await _authService.LogoutAsync(userId, refreshToken);
        }

        ClearRefreshTokenCookie();

        return Ok(new AuthResponse
        {
            Success = true,
            Message = "Logged out successfully"
        });
    }

    [HttpPost("forgot-password")]
    public async Task<ActionResult<AuthResponse>> ForgotPassword([FromBody] ForgotPasswordRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new AuthResponse
            {
                Success = false,
                Message = "Invalid request data",
                Errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList()
            });
        }

        var result = await _authService.ForgotPasswordAsync(request);
        return Ok(result); // Always return OK to prevent email enumeration
    }

    [HttpPost("reset-password")]
    public async Task<ActionResult<AuthResponse>> ResetPassword([FromBody] ResetPasswordRequest request)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(new AuthResponse
            {
                Success = false,
                Message = "Invalid request data",
                Errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList()
            });
        }

        var result = await _authService.ResetPasswordAsync(request);
        
        if (!result.Success)
        {
            return BadRequest(result);
        }

        return Ok(result);
    }

    [HttpGet("user")]
    [Authorize]
    public ActionResult<UserInfo> GetCurrentUser()
    {
        var userInfo = new UserInfo
        {
            Id = User.FindFirst("sub")?.Value ?? "",
            Email = User.FindFirst("email")?.Value ?? "",
            FirstName = User.FindFirst("first_name")?.Value ?? "",
            LastName = User.FindFirst("last_name")?.Value ?? "",
            FavoriteVerse = User.FindFirst("favorite_verse")?.Value,
            SubscriptionTier = Enum.TryParse<SubscriptionTier>(
                User.FindFirst("subscription_tier")?.Value, out var tier) ? tier : SubscriptionTier.Seeker,
            CommunityPoints = int.TryParse(User.FindFirst("community_points")?.Value, out var points) ? points : 0,
            TotalBacktests = int.TryParse(User.FindFirst("total_backtests")?.Value, out var backtests) ? backtests : 0,
            BestSortinoRatio = decimal.TryParse(User.FindFirst("best_sortino")?.Value, out var sortino) ? sortino : 0,
            Roles = User.FindAll("role").Select(c => c.Value).ToList()
        };

        if (DateTime.TryParse(User.FindFirst("subscription_expires")?.Value, out var expires))
        {
            userInfo.SubscriptionExpiresAt = expires;
        }

        return Ok(userInfo);
    }

    [HttpPost("validate-token")]
    public ActionResult<bool> ValidateToken([FromBody] TokenValidationRequest request)
    {
        if (string.IsNullOrEmpty(request.Token))
        {
            return BadRequest(false);
        }

        // For JWT validation, we can decode and check claims without hitting the database
        // This is useful for client-side token validation
        try
        {
            var userId = User.FindFirst("sub")?.Value;
            return Ok(!string.IsNullOrEmpty(userId));
        }
        catch
        {
            return Ok(false);
        }
    }

    private void SetRefreshTokenCookie(string refreshToken)
    {
        var cookieOptions = new CookieOptions
        {
            HttpOnly = true,
            Secure = true, // HTTPS only
            SameSite = SameSiteMode.Strict,
            Expires = DateTime.UtcNow.AddDays(30),
            Path = "/",
            IsEssential = true
        };

        Response.Cookies.Append("refreshToken", refreshToken, cookieOptions);
    }

    private string GetRefreshTokenFromCookie()
    {
        return Request.Cookies["refreshToken"] ?? "";
    }

    private void ClearRefreshTokenCookie()
    {
        var cookieOptions = new CookieOptions
        {
            HttpOnly = true,
            Secure = true,
            SameSite = SameSiteMode.Strict,
            Expires = DateTime.UtcNow.AddDays(-1),
            Path = "/"
        };

        Response.Cookies.Append("refreshToken", "", cookieOptions);
    }
}