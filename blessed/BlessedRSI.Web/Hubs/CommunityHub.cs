using Microsoft.AspNetCore.SignalR;
using Microsoft.AspNetCore.Authorization;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Hubs;

[Authorize]
public class CommunityHub : Hub
{
    private readonly ContentSanitizationService _sanitizationService;
    private readonly ILogger<CommunityHub> _logger;

    public CommunityHub(ContentSanitizationService sanitizationService, ILogger<CommunityHub> logger)
    {
        _sanitizationService = sanitizationService;
        _logger = logger;
    }
    public async Task JoinCommunity()
    {
        await Groups.AddToGroupAsync(Context.ConnectionId, "Community");
        await Clients.Group("Community").SendAsync("UserJoined", Context.User?.Identity?.Name);
    }

    public async Task LeaveCommunity()
    {
        await Groups.RemoveFromGroupAsync(Context.ConnectionId, "Community");
        await Clients.Group("Community").SendAsync("UserLeft", Context.User?.Identity?.Name);
    }

    public async Task SendMessage(string message)
    {
        try
        {
            // Sanitize the message content
            var result = _sanitizationService.SanitizeHtml(message, ContentType.UserComment);
            
            if (!result.IsValid)
            {
                _logger.LogWarning("Malicious message blocked from user {UserId} via SignalR", Context.UserIdentifier);
                await Clients.Caller.SendAsync("MessageBlocked", "Your message contains unsafe content and was not sent.");
                return;
            }

            var sanitizedMessage = result.SanitizedContent;
            var userName = Context.User?.Identity?.Name ?? "Anonymous";

            // Log if content was modified
            if (result.IsModified)
            {
                _logger.LogInformation("SignalR message content sanitized for user {UserId}", Context.UserIdentifier);
            }

            await Clients.Group("Community").SendAsync("ReceiveMessage", userName, sanitizedMessage);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing SignalR message from user {UserId}", Context.UserIdentifier);
            await Clients.Caller.SendAsync("MessageError", "There was an error sending your message.");
        }
    }

    public async Task NotifyNewPost(string postTitle, string userName)
    {
        try
        {
            // Sanitize notification content
            var sanitizedTitle = _sanitizationService.SanitizePlainText(postTitle);
            var sanitizedUserName = _sanitizationService.SanitizePlainText(userName);
            
            await Clients.Group("Community").SendAsync("NewPost", sanitizedTitle, sanitizedUserName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing new post notification");
        }
    }

    public async Task NotifyNewComment(int postId, string userName)
    {
        await Clients.Group("Community").SendAsync("NewComment", postId, userName);
    }

    public async Task NotifyAchievementEarned(string userName, string achievementName)
    {
        await Clients.Group("Community").SendAsync("AchievementEarned", userName, achievementName);
    }

    public override async Task OnConnectedAsync()
    {
        await JoinCommunity();
        await base.OnConnectedAsync();
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        await LeaveCommunity();
        await base.OnDisconnectedAsync(exception);
    }
}