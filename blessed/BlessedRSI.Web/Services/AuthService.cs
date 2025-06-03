using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class AuthService
{
    private readonly UserManager<ApplicationUser> _userManager;
    private readonly SignInManager<ApplicationUser> _signInManager;
    private readonly JwtService _jwtService;
    private readonly ApplicationDbContext _context;
    private readonly ILogger<AuthService> _logger;
    private readonly IHttpContextAccessor _httpContextAccessor;

    public AuthService(
        UserManager<ApplicationUser> userManager,
        SignInManager<ApplicationUser> signInManager,
        JwtService jwtService,
        ApplicationDbContext context,
        ILogger<AuthService> logger,
        IHttpContextAccessor httpContextAccessor)
    {
        _userManager = userManager;
        _signInManager = signInManager;
        _jwtService = jwtService;
        _context = context;
        _logger = logger;
        _httpContextAccessor = httpContextAccessor;
    }

    public async Task<AuthResponse> RegisterAsync(RegisterRequest request)
    {
        try
        {
            // Check if user already exists
            var existingUser = await _userManager.FindByEmailAsync(request.Email);
            if (existingUser != null)
            {
                return new AuthResponse
                {
                    Success = false,
                    Message = "User with this email already exists",
                    Errors = new List<string> { "Email is already registered" }
                };
            }

            // Create new user
            var user = new ApplicationUser
            {
                UserName = request.Email,
                Email = request.Email,
                FirstName = request.FirstName,
                LastName = request.LastName,
                FavoriteVerse = request.FavoriteVerse,
                EmailConfirmed = false, // Require email confirmation in production
                CreatedAt = DateTime.UtcNow,
                SubscriptionTier = SubscriptionTier.Seeker
            };

            var result = await _userManager.CreateAsync(user, request.Password);
            
            if (!result.Succeeded)
            {
                return new AuthResponse
                {
                    Success = false,
                    Message = "Failed to create user account",
                    Errors = result.Errors.Select(e => e.Description).ToList()
                };
            }

            // Add user to default role
            await _userManager.AddToRoleAsync(user, "User");

            // Log security event
            await LogSecurityEventAsync(user.Id, SecurityEventType.Registration, "User account created");

            // For production, you would send email confirmation here
            // For now, we'll auto-confirm for development
            var emailToken = await _userManager.GenerateEmailConfirmationTokenAsync(user);
            await _userManager.ConfirmEmailAsync(user, emailToken);

            // Generate tokens
            var roles = await _userManager.GetRolesAsync(user);
            var accessToken = _jwtService.GenerateAccessToken(user, roles);
            var refreshToken = _jwtService.GenerateRefreshToken();

            // Store refresh token
            await StoreRefreshTokenAsync(user.Id, refreshToken);

            return new AuthResponse
            {
                Success = true,
                Message = "Registration successful",
                AccessToken = accessToken,
                RefreshToken = refreshToken,
                TokenExpiration = _jwtService.GetTokenExpiration(accessToken),
                User = MapToUserInfo(user, roles.ToList())
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during user registration for email {Email}", request.Email);
            return new AuthResponse
            {
                Success = false,
                Message = "An error occurred during registration",
                Errors = new List<string> { "Internal server error" }
            };
        }
    }

    public async Task<AuthResponse> LoginAsync(LoginRequest request)
    {
        try
        {
            var user = await _userManager.FindByEmailAsync(request.Email);
            if (user == null)
            {
                await LogSecurityEventAsync(null, SecurityEventType.Login, 
                    $"Failed login attempt for non-existent email: {request.Email}");
                
                return new AuthResponse
                {
                    Success = false,
                    Message = "Invalid email or password",
                    Errors = new List<string> { "Authentication failed" }
                };
            }

            // Check if account is locked
            if (await _userManager.IsLockedOutAsync(user))
            {
                await LogSecurityEventAsync(user.Id, SecurityEventType.AccountLockout, 
                    "Login attempted on locked account");
                
                return new AuthResponse
                {
                    Success = false,
                    Message = "Account is temporarily locked due to multiple failed login attempts",
                    Errors = new List<string> { "Account locked" }
                };
            }

            var result = await _signInManager.CheckPasswordSignInAsync(user, request.Password, lockoutOnFailure: true);
            
            if (!result.Succeeded)
            {
                string reason = result.IsLockedOut ? "Account locked out" :
                               result.IsNotAllowed ? "Account not allowed" :
                               result.RequiresTwoFactor ? "Two factor required" :
                               "Invalid credentials";

                await LogSecurityEventAsync(user.Id, SecurityEventType.Login, 
                    $"Failed login attempt: {reason}");

                return new AuthResponse
                {
                    Success = false,
                    Message = result.IsLockedOut ? "Account locked due to multiple failed attempts" : "Invalid email or password",
                    Errors = new List<string> { reason }
                };
            }

            // Successful login
            var roles = await _userManager.GetRolesAsync(user);
            var accessToken = _jwtService.GenerateAccessToken(user, roles);
            var refreshToken = _jwtService.GenerateRefreshToken();

            // Store refresh token
            await StoreRefreshTokenAsync(user.Id, refreshToken);

            // Reset failed login attempts
            await _userManager.ResetAccessFailedCountAsync(user);

            await LogSecurityEventAsync(user.Id, SecurityEventType.Login, "Successful login");

            return new AuthResponse
            {
                Success = true,
                Message = "Login successful",
                AccessToken = accessToken,
                RefreshToken = refreshToken,
                TokenExpiration = _jwtService.GetTokenExpiration(accessToken),
                User = MapToUserInfo(user, roles.ToList())
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during login for email {Email}", request.Email);
            return new AuthResponse
            {
                Success = false,
                Message = "An error occurred during login",
                Errors = new List<string> { "Internal server error" }
            };
        }
    }

    public async Task<AuthResponse> RefreshTokenAsync(RefreshTokenRequest request)
    {
        try
        {
            var principal = _jwtService.GetPrincipalFromExpiredToken(request.AccessToken);
            if (principal == null)
            {
                return new AuthResponse
                {
                    Success = false,
                    Message = "Invalid access token",
                    Errors = new List<string> { "Token validation failed" }
                };
            }

            var userId = principal.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (string.IsNullOrEmpty(userId))
            {
                return new AuthResponse
                {
                    Success = false,
                    Message = "Invalid token claims",
                    Errors = new List<string> { "User ID not found in token" }
                };
            }

            var storedRefreshToken = await _context.RefreshTokens
                .Include(rt => rt.User)
                .FirstOrDefaultAsync(rt => rt.Token == request.RefreshToken && rt.UserId == userId);

            if (storedRefreshToken == null || !storedRefreshToken.IsActive)
            {
                await LogSecurityEventAsync(userId, SecurityEventType.SuspiciousActivity, 
                    "Invalid refresh token used");
                
                return new AuthResponse
                {
                    Success = false,
                    Message = "Invalid refresh token",
                    Errors = new List<string> { "Refresh token not found or expired" }
                };
            }

            // Revoke old refresh token
            storedRefreshToken.RevokedAt = DateTime.UtcNow;
            storedRefreshToken.RevokedByIp = GetClientIpAddress();

            var user = storedRefreshToken.User;
            var roles = await _userManager.GetRolesAsync(user);

            // Generate new tokens
            var newAccessToken = _jwtService.GenerateAccessToken(user, roles);
            var newRefreshToken = _jwtService.GenerateRefreshToken();

            // Store new refresh token
            storedRefreshToken.ReplacedBy = newRefreshToken;
            await StoreRefreshTokenAsync(userId, newRefreshToken);

            await _context.SaveChangesAsync();

            await LogSecurityEventAsync(userId, SecurityEventType.TokenRefresh, "Token refreshed successfully");

            return new AuthResponse
            {
                Success = true,
                Message = "Token refreshed successfully",
                AccessToken = newAccessToken,
                RefreshToken = newRefreshToken,
                TokenExpiration = _jwtService.GetTokenExpiration(newAccessToken),
                User = MapToUserInfo(user, roles.ToList())
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during token refresh");
            return new AuthResponse
            {
                Success = false,
                Message = "An error occurred during token refresh",
                Errors = new List<string> { "Internal server error" }
            };
        }
    }

    public async Task<bool> LogoutAsync(string userId, string refreshToken)
    {
        try
        {
            // Revoke refresh token
            var token = await _context.RefreshTokens
                .FirstOrDefaultAsync(rt => rt.Token == refreshToken && rt.UserId == userId);

            if (token != null)
            {
                token.RevokedAt = DateTime.UtcNow;
                token.RevokedByIp = GetClientIpAddress();
                await _context.SaveChangesAsync();
            }

            await LogSecurityEventAsync(userId, SecurityEventType.Logout, "User logged out");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during logout for user {UserId}", userId);
            return false;
        }
    }

    public async Task<AuthResponse> ForgotPasswordAsync(ForgotPasswordRequest request)
    {
        try
        {
            var user = await _userManager.FindByEmailAsync(request.Email);
            if (user == null)
            {
                // Don't reveal that user doesn't exist
                return new AuthResponse
                {
                    Success = true,
                    Message = "If an account with that email exists, a password reset link has been sent"
                };
            }

            var token = await _userManager.GeneratePasswordResetTokenAsync(user);
            
            // In production, send email with reset link
            // For now, log the token (remove in production)
            _logger.LogInformation("Password reset token for {Email}: {Token}", request.Email, token);

            await LogSecurityEventAsync(user.Id, SecurityEventType.PasswordReset, "Password reset requested");

            return new AuthResponse
            {
                Success = true,
                Message = "If an account with that email exists, a password reset link has been sent"
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during password reset request for {Email}", request.Email);
            return new AuthResponse
            {
                Success = false,
                Message = "An error occurred while processing your request",
                Errors = new List<string> { "Internal server error" }
            };
        }
    }

    public async Task<AuthResponse> ResetPasswordAsync(ResetPasswordRequest request)
    {
        try
        {
            var user = await _userManager.FindByEmailAsync(request.Email);
            if (user == null)
            {
                return new AuthResponse
                {
                    Success = false,
                    Message = "Invalid reset token",
                    Errors = new List<string> { "User not found" }
                };
            }

            var result = await _userManager.ResetPasswordAsync(user, request.Token, request.NewPassword);
            
            if (!result.Succeeded)
            {
                return new AuthResponse
                {
                    Success = false,
                    Message = "Failed to reset password",
                    Errors = result.Errors.Select(e => e.Description).ToList()
                };
            }

            // Revoke all refresh tokens for security
            await RevokeAllRefreshTokensAsync(user.Id);

            await LogSecurityEventAsync(user.Id, SecurityEventType.PasswordReset, "Password reset completed");

            return new AuthResponse
            {
                Success = true,
                Message = "Password has been reset successfully"
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during password reset for {Email}", request.Email);
            return new AuthResponse
            {
                Success = false,
                Message = "An error occurred while resetting your password",
                Errors = new List<string> { "Internal server error" }
            };
        }
    }

    private async Task StoreRefreshTokenAsync(string userId, string refreshToken)
    {
        var token = new RefreshToken
        {
            Token = refreshToken,
            UserId = userId,
            ExpiresAt = DateTime.UtcNow.AddDays(30), // 30 day expiration
            CreatedByIp = GetClientIpAddress()
        };

        _context.RefreshTokens.Add(token);
        await _context.SaveChangesAsync();

        // Clean up old tokens
        await CleanupExpiredTokensAsync(userId);
    }

    private async Task CleanupExpiredTokensAsync(string userId)
    {
        var expiredTokens = await _context.RefreshTokens
            .Where(rt => rt.UserId == userId && (rt.IsExpired || rt.IsRevoked))
            .Where(rt => rt.CreatedAt < DateTime.UtcNow.AddDays(-60)) // Keep for 60 days for audit
            .ToListAsync();

        _context.RefreshTokens.RemoveRange(expiredTokens);
        await _context.SaveChangesAsync();
    }

    private async Task RevokeAllRefreshTokensAsync(string userId)
    {
        var tokens = await _context.RefreshTokens
            .Where(rt => rt.UserId == userId && rt.IsActive)
            .ToListAsync();

        foreach (var token in tokens)
        {
            token.RevokedAt = DateTime.UtcNow;
            token.RevokedByIp = GetClientIpAddress();
        }

        await _context.SaveChangesAsync();
    }

    private async Task LogSecurityEventAsync(string? userId, SecurityEventType eventType, string description)
    {
        try
        {
            var securityEvent = new SecurityEvent
            {
                UserId = userId ?? "",
                EventType = eventType,
                Description = description,
                IpAddress = GetClientIpAddress(),
                UserAgent = _httpContextAccessor.HttpContext?.Request.Headers["User-Agent"].ToString()
            };

            _context.SecurityEvents.Add(securityEvent);
            await _context.SaveChangesAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to log security event");
        }
    }

    private string? GetClientIpAddress()
    {
        var context = _httpContextAccessor.HttpContext;
        if (context == null) return null;

        var ipAddress = context.Request.Headers["X-Forwarded-For"].FirstOrDefault();
        if (string.IsNullOrEmpty(ipAddress))
        {
            ipAddress = context.Request.Headers["X-Real-IP"].FirstOrDefault();
        }
        if (string.IsNullOrEmpty(ipAddress))
        {
            ipAddress = context.Connection.RemoteIpAddress?.ToString();
        }

        return ipAddress;
    }

    private static UserInfo MapToUserInfo(ApplicationUser user, List<string> roles)
    {
        return new UserInfo
        {
            Id = user.Id,
            Email = user.Email ?? "",
            FirstName = user.FirstName ?? "",
            LastName = user.LastName ?? "",
            FavoriteVerse = user.FavoriteVerse,
            SubscriptionTier = user.SubscriptionTier,
            SubscriptionExpiresAt = user.SubscriptionExpiresAt,
            CommunityPoints = user.CommunityPoints,
            TotalBacktests = user.TotalBacktests,
            BestSortinoRatio = user.BestSortinoRatio,
            Roles = roles
        };
    }
}