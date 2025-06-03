using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.AspNetCore.Components.Server.ProtectedBrowserStorage;
using System.Security.Claims;
using System.Text.Json;

namespace BlessedRSI.Web.Services;

public class CustomAuthenticationStateProvider : AuthenticationStateProvider
{
    private readonly ProtectedLocalStorage _localStorage;
    private readonly JwtService _jwtService;
    private readonly ILogger<CustomAuthenticationStateProvider> _logger;
    private readonly IHttpContextAccessor _httpContextAccessor;

    public CustomAuthenticationStateProvider(
        ProtectedLocalStorage localStorage,
        JwtService jwtService,
        ILogger<CustomAuthenticationStateProvider> logger,
        IHttpContextAccessor httpContextAccessor)
    {
        _localStorage = localStorage;
        _jwtService = jwtService;
        _logger = logger;
        _httpContextAccessor = httpContextAccessor;
    }

    public override async Task<AuthenticationState> GetAuthenticationStateAsync()
    {
        try
        {
            // First try to get JWT from HTTP context (for API calls)
            var httpContext = _httpContextAccessor.HttpContext;
            if (httpContext != null)
            {
                var authHeader = httpContext.Request.Headers["Authorization"].FirstOrDefault();
                if (!string.IsNullOrEmpty(authHeader) && authHeader.StartsWith("Bearer "))
                {
                    var token = authHeader.Substring("Bearer ".Length).Trim();
                    if (_jwtService.ValidateToken(token))
                    {
                        var principal = _jwtService.GetPrincipalFromExpiredToken(token);
                        if (principal != null)
                        {
                            return new AuthenticationState(principal);
                        }
                    }
                }
            }

            // For Blazor Server, try to get token from protected browser storage
            try
            {
                var tokenResult = await _localStorage.GetAsync<string>("accessToken");
                if (tokenResult.Success && !string.IsNullOrEmpty(tokenResult.Value))
                {
                    if (_jwtService.ValidateToken(tokenResult.Value))
                    {
                        var principal = CreateClaimsPrincipalFromToken(tokenResult.Value);
                        if (principal != null)
                        {
                            return new AuthenticationState(principal);
                        }
                    }
                    else
                    {
                        // Token is invalid or expired, try to refresh
                        await TryRefreshTokenAsync();
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to retrieve token from local storage");
            }

            return new AuthenticationState(new ClaimsPrincipal(new ClaimsIdentity()));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting authentication state");
            return new AuthenticationState(new ClaimsPrincipal(new ClaimsIdentity()));
        }
    }

    public async Task MarkUserAsAuthenticatedAsync(string token)
    {
        try
        {
            await _localStorage.SetAsync("accessToken", token);
            
            var principal = CreateClaimsPrincipalFromToken(token);
            if (principal != null)
            {
                NotifyAuthenticationStateChanged(Task.FromResult(new AuthenticationState(principal)));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error marking user as authenticated");
        }
    }

    public async Task MarkUserAsLoggedOutAsync()
    {
        try
        {
            await _localStorage.DeleteAsync("accessToken");
            var anonymous = new ClaimsPrincipal(new ClaimsIdentity());
            NotifyAuthenticationStateChanged(Task.FromResult(new AuthenticationState(anonymous)));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error marking user as logged out");
        }
    }

    private ClaimsPrincipal? CreateClaimsPrincipalFromToken(string token)
    {
        try
        {
            var handler = new System.IdentityModel.Tokens.Jwt.JwtSecurityTokenHandler();
            var jwt = handler.ReadJwtToken(token);

            var claims = new List<Claim>();

            // Extract claims from JWT
            foreach (var claim in jwt.Claims)
            {
                // Map JWT claim types to ASP.NET Core claim types
                var claimType = claim.Type switch
                {
                    "sub" => ClaimTypes.NameIdentifier,
                    "email" => ClaimTypes.Email,
                    "name" => ClaimTypes.Name,
                    "role" => ClaimTypes.Role,
                    _ => claim.Type
                };

                claims.Add(new Claim(claimType, claim.Value));
            }

            var identity = new ClaimsIdentity(claims, "jwt");
            return new ClaimsPrincipal(identity);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to create claims principal from token");
            return null;
        }
    }

    private async Task TryRefreshTokenAsync()
    {
        try
        {
            // This would call the refresh token endpoint
            // For now, just clear the invalid token
            await _localStorage.DeleteAsync("accessToken");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to refresh token");
        }
    }
}